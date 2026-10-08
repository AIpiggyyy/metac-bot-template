"""
Ensemble forecasting bot (Alex's Money project, Oct 2026).

Built on the official FallTemplateBot2026 (main.py) so all question-type handling, parsing and submission
stay the battle-tested template code. What this adds:
  1. A multi-model ensemble: each forecast call rotates through several frontier models (newest available,
     picked automatically at start-up), and the template's median aggregation combines them.
     Spring-2026 results: multi-model pipelines took the top places; the best single-model baseline was 18th.
  2. Combined research: AskNews news summaries (if keys) + a web-search LLM (if available), merged.
  3. Fallback: if an ensemble model errors, the call is retried on the next model so one broken
     provider can't sink a question.
  4. Optional mild extremizing of the aggregated binary forecast (off by default; enable only after
     backtesting on resolved questions).

Configuration (GitHub repository *variables* or secrets; all optional):
  ENSEMBLE_MODELS       comma list of litellm model ids to use instead of auto-pick
  PREDICTIONS_PER_Q     forecasts per question (default 10 = two passes over the 5-model rotation)
  RESEARCH_MODEL        litellm id of a web-search model (default: auto)
  EXTREMIZE             logit multiplier for binary aggregate (default 1.0 = off)
  DRY_RUN               "1" = don't publish to Metaculus
  TOURNAMENT_ID         override the seasonal tournament (id or slug, e.g. fall-futureeval-2026)
  SPOT_TOURNAMENTS      comma list of spot-scored tournaments to also forecast in tournament mode
                        (default: market-pulse-26q4; "none" to disable). Only the forecast standing at close
                        counts there, so questions are forecast once when first seen and re-forecast in the last
                        SPOT_WINDOW_H hours before close (at most once per SPOT_REFRESH_H hours).

Run:  python ensemble_bot.py --mode tournament | test_questions | metaculus_cup
"""
import argparse
import asyncio
import datetime as dt
import itertools
import logging
import math
import os
from typing import Literal

import requests

from main import (  # the official template (keeps prompts, parsing, submission)
    FallTemplateBot2026,
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
)
from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
)

logger = logging.getLogger("ensemble_bot")

# ----------------------------------------------------------------------------- model selection

# (family label, id must start with, id must contain one of, weight in the rotation)
FAMILIES = [
    ("openai", "openai/gpt-5", [""], 2),          # GPT-5.x did best in Spring 2026 -> double weight
    ("google", "google/gemini-", ["pro"], 1),
    ("anthropic", "anthropic/claude-", ["opus", "sonnet"], 1),
    ("xai", "x-ai/grok-", [""], 1),
]
EXCLUDE = ["mini", "nano", "flash", "lite", "fast", "image", "audio", "codex", "search", "deep-research", "online",
           ":free", "realtime", "oss", "chat", "vision", "embed", "tts", "exp"]   # matched on the part AFTER the family prefix
MAX_COMPLETION_USD_PER_TOKEN = 80e-6     # skip ultra-premium tiers (e.g. $120+/M output "pro" models)


def _env(name: str) -> bool:
    v = os.getenv(name)
    return bool(v and v.strip() and v.strip() not in {"REPLACE_ME", "1234567890"})


def pick_openrouter_models() -> list[str]:
    """Newest reasonably-priced flagship per family from OpenRouter's public model list."""
    try:
        data = requests.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    except Exception as e:  # network hiccup -> conservative fixed list
        logger.warning(f"OpenRouter model list unavailable ({e}); using fallback ids")
        return ["openrouter/openai/gpt-5", "openrouter/google/gemini-2.5-pro",
                "openrouter/anthropic/claude-sonnet-4.5", "openrouter/openai/gpt-5"]
    chosen: list[str] = []
    for label, prefix, needs, weight in FAMILIES:
        cands = []
        for m in data:
            mid = m.get("id", "")
            if not mid.startswith(prefix):
                continue
            tail = mid[len(prefix):]                       # e.g. "3-pro" for google/gemini-3-pro ("gemini" contains "mini")
            if any(x in tail for x in EXCLUDE) or not any(n in tail for n in needs):
                continue
            try:
                price = float((m.get("pricing") or {}).get("completion") or 0)
            except ValueError:
                price = 0
            if price <= 0 or price > MAX_COMPLETION_USD_PER_TOKEN:
                continue
            cands.append((m.get("created", 0), mid))
        if cands:
            best = max(cands)[1]
            chosen += [f"openrouter/{best}"] * weight
            logger.info(f"ensemble: {label} -> {best} (x{weight})")
        else:
            logger.warning(f"ensemble: no model found for family {label}")
    return chosen


def pick_anthropic_models() -> list[str]:
    """Anthropic-only mode (smoke tests on Alex's small API credit): newest Sonnet."""
    try:
        r = requests.get("https://api.anthropic.com/v1/models", timeout=30,
                         headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
        ids = [m["id"] for m in r.json().get("data", [])]
        sonnets = [i for i in ids if "sonnet" in i]
        if sonnets:
            return [f"anthropic/{sonnets[0]}"]          # API lists newest first
    except Exception as e:
        logger.warning(f"Anthropic model list unavailable ({e})")
    return ["anthropic/claude-sonnet-4-5"]


def ensemble_model_ids() -> list[str]:
    if _env("ENSEMBLE_MODELS"):
        return [m.strip() for m in os.environ["ENSEMBLE_MODELS"].split(",") if m.strip()]
    if _env("OPENROUTER_API_KEY"):
        return pick_openrouter_models()
    if _env("ANTHROPIC_API_KEY"):
        return pick_anthropic_models()
    if _env("METACULUS_TOKEN"):                          # Metaculus LLM proxy (credits granted to the token)
        return ["metaculus/gpt-4o", "metaculus/anthropic/claude-3-5-sonnet-20241022"]
    raise RuntimeError("No LLM key configured")


NO_TEMPERATURE = ("anthropic/", "claude", "openai/gpt-5", "openai/o", "x-ai/grok")   # these reject/deprecate temperature


def _strip_temperature(llm: GeneralLlm) -> GeneralLlm:
    """Newer reasoning models (e.g. Claude 5.x, GPT-5.x) return 400 'temperature is deprecated'. The pinned
    forecasting-tools (0.2.92) doesn't auto-drop it, so never send it for those models."""
    if any(k in llm.model for k in NO_TEMPERATURE):
        llm.litellm_kwargs.pop("temperature", None)
    return llm


def make_llm(model_id: str, temperature: float | None = 0.3, timeout: int = 240) -> GeneralLlm:
    kw = dict(model=model_id, temperature=temperature, timeout=timeout, allowed_tries=2)
    if "openai/gpt-5" in model_id or "x-ai/grok" in model_id:
        kw["reasoning_effort"] = "high"                 # high-reasoning variants beat standard twins 8/8 (Spring 2026)
    return _strip_temperature(GeneralLlm(**kw))


def parser_model_id(ensemble_ids: list[str]) -> str:
    """Cheap model that turns free-text forecasts into structured numbers."""
    if _env("OPENROUTER_API_KEY"):
        return "openrouter/openai/gpt-4o-mini"
    if _env("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if _env("ANTHROPIC_API_KEY"):
        return ensemble_ids[0]            # Anthropic-only mode: reuse the picked (current) Claude model
    return "metaculus/gpt-4o-mini"


def research_model_id() -> str | None:
    if _env("RESEARCH_MODEL"):
        return os.environ["RESEARCH_MODEL"]
    if _env("PERPLEXITY_API_KEY"):
        return "perplexity/sonar-pro"
    if _env("OPENROUTER_API_KEY"):
        return "openrouter/perplexity/sonar-pro"
    return None


# ----------------------------------------------------------------------------- fallback wrapper

class FallbackLlm:
    """Duck-types GeneralLlm.invoke: try the assigned model, then the others in order."""

    def __init__(self, primary: GeneralLlm, others: list[GeneralLlm]):
        self.primary, self.others = primary, others
        self.model = primary.model

    async def invoke(self, prompt):
        errors = []
        for llm in [self.primary] + [o for o in self.others if o.model != self.primary.model]:
            try:
                out = await llm.invoke(prompt)
                if out and str(out).strip():
                    if llm is not self.primary:
                        logger.warning(f"fallback used: {self.primary.model} -> {llm.model}")
                    return out
                errors.append(f"{llm.model}: empty")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{llm.model}: {str(e)[:200]}")
        raise RuntimeError("all ensemble models failed: " + " | ".join(errors))


# ----------------------------------------------------------------------------- the bot

class EnsembleBot(FallTemplateBot2026):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, ensemble_ids: list[str], research_id: str | None, extremize: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble = [make_llm(m) for m in ensemble_ids]
        uniq = list({l.model: l for l in self.ensemble}.values())
        self._rotation = itertools.cycle(self.ensemble)
        self._uniq = uniq
        self.research_id = research_id
        self.extremize = extremize

    def get_llm(self, purpose: str = "default", guarantee_type=None):
        if purpose == "default":
            nxt = next(self._rotation)
            return FallbackLlm(nxt, self._uniq)
        return super().get_llm(purpose, guarantee_type)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            parts = []
            if (_env("ASKNEWS_CLIENT_ID") and _env("ASKNEWS_SECRET")) or _env("ASKNEWS_API_KEY"):
                try:
                    news = await AskNewsSearcher().call_preconfigured_version(
                        "asknews/news-summaries", question.question_text)
                    parts.append("## News (AskNews)\n" + news)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"AskNews failed: {e}")
            if self.research_id:
                try:
                    prompt = self._get_research_prompt(question, self.research_id) + finance_hint(question)
                    web = await make_llm(self.research_id, temperature=0.1, timeout=180).invoke(prompt)
                    parts.append("## Web research\n" + web)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"web research failed: {e}")
            if not parts:   # last resort: ask the first ensemble model what it knows
                try:
                    prompt = self._get_research_prompt(question, "llm") + finance_hint(question)
                    parts.append("## Background (model knowledge, no live search)\n"
                                 + await self._uniq[0].invoke(prompt))
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"fallback research failed: {e}")
            research = "\n\n".join(parts)
            logger.info(f"Research for {question.page_url}: {len(research)} chars from {len(parts)} source(s)")
            return research

    async def _aggregate_predictions(self, predictions, question):
        agg = await super()._aggregate_predictions(predictions, question)
        if isinstance(question, BinaryQuestion) and self.extremize != 1.0:
            p = min(max(float(agg), 0.01), 0.99)
            z = math.log(p / (1 - p)) * self.extremize
            agg = min(max(1 / (1 + math.exp(-z)), 0.01), 0.99)
        return agg


# ----------------------------------------------------------------------------- spot-scored tournaments

FINANCE_HINT = (
    "\n\nThis is a company-financials question. Check the exact resolution basis in the question "
    "(e.g. GAAP diluted EPS vs adjusted/non-GAAP; total revenue; the company's own guidance figure). Report: "
    "(1) the latest analyst consensus for the exact metric and whether it is GAAP or adjusted, with date and "
    "source; (2) the company's own guidance for the period; (3) the last 8 quarters of the reported metric and "
    "how far each beat or missed consensus; (4) known one-off GAAP items this quarter (stock comp, impairments, "
    "investment gains/losses, tax items) that make GAAP differ from adjusted; (5) the confirmed report date.")
FINANCE_WORDS = ("earnings per share", "revenue", "guidance", "gross margin", "operating expenses")


def finance_hint(question: MetaculusQuestion) -> str:
    t = (question.question_text or "").lower()
    return FINANCE_HINT if any(w in t for w in FINANCE_WORDS) else ""


def spot_due(q: MetaculusQuestion, now: dt.datetime) -> bool:
    """Forecast if never forecast, or if inside the pre-close window and the last forecast is stale."""
    if not q.already_forecasted:
        return True
    if q.close_time is None:
        return False
    close = q.close_time if q.close_time.tzinfo else q.close_time.replace(tzinfo=dt.timezone.utc)
    hours_left = (close - now).total_seconds() / 3600
    if hours_left > float(os.getenv("SPOT_WINDOW_H") or "30") or hours_left < 0.5:
        return False
    last = q.timestamp_of_my_last_forecast
    if last is None:
        return True
    last = last if last.tzinfo else last.replace(tzinfo=dt.timezone.utc)
    return (now - last).total_seconds() / 3600 >= float(os.getenv("SPOT_REFRESH_H") or "20")


def spot_tournaments() -> list[str]:
    v = (os.getenv("SPOT_TOURNAMENTS") or "market-pulse-26q4").strip()
    return [] if v.lower() == "none" else [x.strip() for x in v.split(",") if x.strip()]


# ----------------------------------------------------------------------------- entry point

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tournament", "metaculus_cup", "test_questions"], default="tournament")
    args = ap.parse_args()
    mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish = (os.getenv("DRY_RUN") or "0") != "1"
    print_startup_banner(mode, will_publish=publish)

    ids = ensemble_model_ids()
    n_pred = int(os.getenv("PREDICTIONS_PER_Q") or "10")
    if len(set(ids)) == 1:
        n_pred = min(n_pred, 5)            # single-model mode (smoke test): keep it cheap
    logger.info(f"ensemble models: {ids}; predictions per question: {n_pred}")

    bot = EnsembleBot(
        research_reports_per_question=1,
        predictions_per_research_report=n_pred,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={"default": ids[0], "parser": make_llm(parser_model_id(ids), temperature=0.0, timeout=120),
              "summarizer": make_llm(parser_model_id(ids), temperature=0.0, timeout=120),
              "researcher": research_model_id() or "no_research"},
        ensemble_ids=ids,
        research_id=research_model_id(),
        extremize=float(os.getenv("EXTREMIZE") or "1.0"),
    )

    client = MetaculusClient()
    urls = {"tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
            "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
            "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/"}
    if mode == "tournament":
        # forecasting-tools 0.2.92 (pinned in the template's poetry.lock) still points at the closed SUMMER tournament
        # (33022). The workflows upgrade to >=0.3.4 (Fall = 33121); TOURNAMENT_ID can override either way.
        tid = os.getenv("TOURNAMENT_ID") or client.CURRENT_AI_COMPETITION_ID
        if str(tid) == "33022":
            logger.warning("library points at the closed Summer-2026 tournament; using fall-futureeval-2026")
            tid = "fall-futureeval-2026"
        logger.info(f"tournament: {tid}")
        reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
        reports += asyncio.run(bot.forecast_on_tournament(client.CURRENT_MINIBENCH_ID, return_exceptions=True))
        for spot_tid in spot_tournaments():
            try:
                qs = client.get_all_open_questions_from_tournament(spot_tid)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"spot tournament {spot_tid}: could not list questions ({e})")
                continue
            now = dt.datetime.now(dt.timezone.utc)
            due = [q for q in qs if spot_due(q, now)]
            logger.info(f"spot tournament {spot_tid}: {len(qs)} open, {len(due)} due now")
            if due:
                bot.skip_previously_forecasted_questions = False
                reports += asyncio.run(bot.forecast_questions(due, return_exceptions=True))
                bot.skip_previously_forecasted_questions = True
    elif mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament(client.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
    else:
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))

    bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=urls.get(mode))
