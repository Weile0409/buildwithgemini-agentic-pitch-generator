# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
/**
 * @file agent.py
 * @description Multi-agent ADK workflow definition and service-injected execution
 *   engine for the Agentic Pitch Generator.
 *
 * Why: Defines the Google ADK specialist agent team (`creative_director`,
 * `brand_strategist`, `copywriter`, `visual_director`), structured payload contracts,
 * `LoopGuard` DAG validation, `assemble = JoinNode(name="assemble")`, `package`,
 * `root_agent`, and `app` running on Gemini Enterprise Agent Platform (`gemini-3.8-flash`)
 * while providing lightweight stdlib fallback shims when `google-adk` / `google-genai`
 * wheels are not installed locally. Also exposes `run_pitch_workflow` and
 * `resume_pitch_workflow` wired through `ServiceContainer` so local runs use live
 * Cloud models and offline test suites execute deterministically.
 */
"""

from __future__ import annotations

# ==============================================================================
# LAB GUIDEPOST INDEX (`pitch_generator/agent.py`)
#   - Step 1a: `# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]`
#   - Step 1b: `# [Guidepost — Step 1b: Skill Evaluation Harness]`
#   - Step 1c: `# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]`
#   - Step 3a: `# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]`
#   - Step 3c: `# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]`
#   - Step 4a: `# [Guidepost — Step 4a: Tokenomics & Context Optimization]`
# (Tip: All specialist agents, workflow edges, and lab guideposts are in the top
#  ~320 lines of this file—read lines 1–350 in a single `view_file` call!)
# ==============================================================================

from collections.abc import AsyncIterator, Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import time
from typing import Any

from pitch_generator.app_utils.adk_primitives import (
    Agent,
    App,
    ArtDirectionPayload,
    CampaignConcept,
    CircularLoopError,
    ConceptPayload,
    Context,
    CopyPayload,
    Event,
    Gemini,
    JoinNode,
    LoopGuard,
    PayloadValidationError,
    PitchPackage,
    RemoteA2aAgent,
    RequestInput,
    VisualPayload,
    Workflow,
    _DualSyncAsyncEventStream,
    _GenAITypesShim,
    _SessionData,
    _extract_json_Candidate,
    _key_visual,
    _node_name,
    _normalize_directed_edges,
    extract_json_payload,
    node,
    parse_json_payload,
    parse_payload,
    strip_markdown_fences,
    types,
)
from pitch_generator.app_utils.a2a import (
    AGENT_CARD_WELL_KNOWN_PATH,
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
    AgentCardBuilder,
    DefaultRequestHandler,
    include_artifacts_in_a2a_event_interceptor,
)
from pitch_generator.app_utils.services import (
    ServiceContainer,
    get_default_services,
)
from pitch_generator.config import (
    PitchConfig,
    get_config,
)

MODEL: str = get_config().flash_model
IMAGE_MODEL: str = get_config().image_model


def _model(config: PitchConfig | None = None) -> Gemini:
    """
    /**
     * Construct the default `Gemini` model configuration (`gemini-3.8-flash`) for
     * specialist agents.
     *
     * Why: Centralizes `location="global"` and `HttpRetryOptions(attempts=3)` so every
     * specialist agent shares resilient retry settings on Gemini Enterprise Agent Platform.
     *
     * @param config Optional `PitchConfig` override.
     * @return Configured `Gemini` model descriptor.
     */
    """
    cfg = config or get_config()
    return Gemini(
        model=cfg.flash_model,
        client_kwargs={"location": cfg.location},
        retry_options=types.HttpRetryOptions(attempts=3),
    )


_build_model = _model


async def generate_key_visual(
    art_direction: str,
    tool_context: Context | None = None,
    *,
    simulate_no_image: bool = False,
) -> dict[str, Any]:
    """
    /**
     * Generate a 16:9 key visual PNG from art direction notes and save it to session artifacts.
     *
     * Why: Attached to `visual_director` (`tools=[generate_key_visual]`) so the Visual
     * Director specialist can synthesize a key visual image via the configured
     * `image_model` (`gemini-nano-banana-2.1`) and store it in the session artifact store.
     *
     * @param art_direction Art direction prompt describing palette, lighting, composition, and subject.
     * @param tool_context Optional ADK `Context` / `ToolContext` for saving the image artifact.
     * @param simulate_no_image If True, simulates an empty image response to test error handling.
     * @return Metadata dictionary with `filename`, `version`, `bytes`, and `mime_type`.
     */
    """
    import mimetypes

    ctx = tool_context if tool_context is not None else Context()
    cfg = ctx.services.config if hasattr(ctx, "services") and ctx.services else get_config()
    if (
        simulate_no_image
        or getattr(ctx, "simulate_no_image", False)
        or not isinstance(art_direction, str)
        or not art_direction.strip()
    ):
        raise ValueError(f"{cfg.image_model} returned no image for: {str(art_direction)[:120]}")

    image_bytes, mime_type = ctx.services.llm.generate_image(
        art_direction.strip(),
        model=cfg.image_model,
    )
    if not image_bytes:
        raise ValueError(f"{cfg.image_model} returned no image for: {art_direction[:120]}")

    ext = mimetypes.guess_extension(mime_type or "") or ".png"
    filename = f"key_visual{ext}"
    image_part = types.Part(
        inline_data=types.Blob(data=image_bytes, mime_type=mime_type)
    )
    version = await ctx.save_artifact(filename, image_part)
    ctx.session.events.append(
        Event(
            author="visual_director",
            content=types.Content(role="model", parts=[image_part]),
        )
    )
    return {
        "filename": filename,
        "version": version,
        "bytes": len(image_bytes),
        "mime_type": mime_type,
    }


def build_specialist_team(config: PitchConfig | None = None) -> dict[str, Agent]:
    """
    /**
     * Instantiate the four domain specialist agents for the Pitch Generator team.
     *
     * Why: Provides a deterministic factory that binds each specialist role
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) to
     * a focused single-responsibility instruction and a unique `output_key` so
     * downstream graph nodes can read each specialist's output without collision.
     *
     * @param config Optional `PitchConfig` used to resolve model settings.
     * @return Dictionary mapping role name to its configured `Agent` instance.
     */
    """
    model = _model(config)
    brand_skill = load_brand_skill()
    brand_toolset = SkillToolset([brand_skill])
    visual_instruction = (
        "You are the Visual Director. Follow this mandatory procedure:\n"
        "1. First call `load_skill` with `skill_name=\"brand-guidelines\"` to load house visual guidelines.\n"
        "2. Write the art direction incorporating the loaded brand guidelines.\n"
        "3. Call `generate_key_visual` with your art direction prompt.\n"
        "4. Output ONLY the written art direction text."
    )
    return {
        "creative_director": Agent(
            name="creative_director",
            model=model,
            description="Turns a raw campaign topic into a punchy concept line and strategic rationale.",
            instruction=(
                "You are the Creative Director. Turn the campaign idea you are given into "
                "ONE punchy campaign concept line accompanied by a clear rationale explanation."
            ),
            output_key="creative_director",
        ),
        "copywriter": Agent(
            name="copywriter",
            model=model,
            description="Writes a concise, high-impact social media caption under 25 words.",
            instruction=(
                "You are the Copywriter. Write ONE short social caption for the campaign "
                "concept you are given. Strictly under 25 words with an energetic call to action."
            ),
            output_key="copywriter",
        ),
        "brand_strategist": Agent(
            name="brand_strategist",
            model=model,
            description="Defines campaign positioning, target audience, and brand tone for the campaign concept.",
            instruction=(
                "You are the Brand Strategist. Define the campaign positioning, "
                "target audience, and brand tone for the campaign concept."
            ),
            output_key="brand_strategist",
        ),
        "visual_director": Agent(
            name="visual_director",
            model=model,
            description="Translates the campaign concept into art direction and key visuals.",
            instruction=visual_instruction,
            tools=[brand_toolset, generate_key_visual],
            output_key="visual_director",
        ),
    }


@dataclass
class Skill:
    """
    /**
     * Data representation of an on-demand agent skill.
     *
     * @param name Unique skill name.
     * @param description Brief description of the skill and its trigger context.
     * @param body Markdown content of the skill.
     * @param path Optional filesystem path to the skill source file.
     */
    """
    name: str
    description: str
    body: str
    path: Path | None = None


class SkillToolset:
    """
    /**
     * Toolset providing on-demand skill loading for agents.
     *
     * Why: Wraps loaded skills and exposes a `load_skill(skill_name)` tool callable
     * by agents to retrieve domain-specific guidelines dynamically into context.
     */
    """

    def __init__(self, skills: Sequence[Skill]) -> None:
        self.skills: dict[str, Skill] = {s.name: s for s in skills}
        self.loaded_skills: list[str] = []

    def load_skill(self, skill_name: str) -> str:
        """
        /**
         * Load the body text of a named skill into context.
         *
         * @param skill_name The name of the skill to load (e.g. "brand-guidelines").
         * @return The Markdown body of the skill.
         */
        """
        if not isinstance(skill_name, str) or not skill_name.strip():
            raise ValueError("Skill name must be a non-empty string.")
        clean_name = skill_name.strip()
        if clean_name not in self.skills:
            raise KeyError(f"Skill {skill_name!r} not found. Available: {list(self.skills.keys())}")
        if clean_name not in self.loaded_skills:
            self.loaded_skills.append(clean_name)
        return self.skills[clean_name].body

    def get_tools(self) -> list[Any]:
        return [self.load_skill]


def load_skill_from_dir(skill_dir: str | Path) -> Skill:
    """
    /**
     * Parse and load a Skill from a directory containing a SKILL.md file.
     *
     * @param skill_dir Directory containing SKILL.md.
     * @return Instantiated Skill object.
     */
    """
    dir_path = Path(skill_dir).resolve()
    skill_file = dir_path / "SKILL.md" if dir_path.is_dir() else dir_path
    if not skill_file.is_file():
        raise FileNotFoundError(f"Skill file not found at {skill_file}")

    content = skill_file.read_text(encoding="utf-8")
    name = dir_path.name
    desc = ""
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            body = parts[2].strip()
            for line in frontmatter.strip().splitlines():
                if line.startswith("name:"):
                    name = line.split("name:", 1)[1].strip()
                elif line.startswith("description:"):
                    desc = line.split("description:", 1)[1].strip()

    return Skill(name=name, description=desc, body=body, path=skill_file)


def load_brand_skill() -> Skill:
    """
    /**
     * Helper to load the canonical brand-guidelines skill.
     *
     * @return Instantiated brand-guidelines Skill.
     */
    """
    skill_path = Path(__file__).resolve().parent / "skills" / "brand-guidelines"
    return load_skill_from_dir(skill_path)


create_specialist_agents = build_specialist_team
get_specialist_agents = build_specialist_team

brand_guidelines_skill: Skill = load_brand_skill()
SPECIALIST_AGENTS: dict[str, Agent] = build_specialist_team()

creative_director: Agent = SPECIALIST_AGENTS["creative_director"]
copywriter: Agent = SPECIALIST_AGENTS["copywriter"]
brand_strategist: Agent = SPECIALIST_AGENTS["brand_strategist"]
visual_director: Agent = SPECIALIST_AGENTS["visual_director"]


def get_specialist_agent(role: str) -> Agent:
    """
    /**
     * Look up a single specialist `Agent` by role identifier.
     *
     * Why: Fails fast with a descriptive `KeyError` when an invalid or unknown
     * specialist role is requested instead of silently returning `None`.
     *
     * @param role Specialist role name (e.g., `"creative_director"`, `"copywriter"`).
     * @return Matching `Agent` instance.
     */
    """
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Specialist role name must be a non-empty string")
    clean_role = role.strip()
    if clean_role == "market_researcher":
        return SPECIALIST_AGENTS["brand_strategist"]
    if clean_role not in SPECIALIST_AGENTS:
        raise KeyError(
            f"Unknown specialist role {role!r}. Expected one of {tuple(SPECIALIST_AGENTS.keys())}."
        )
    return SPECIALIST_AGENTS[clean_role]


def run_specialist_team(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    config: PitchConfig | None = None,
) -> dict[str, str]:
    """
    /**
     * Execute all four specialist agents sequentially/in-parallel against a campaign brief.
     *
     * Why: Allows learners and automated tests to verify that all four specialist roles
     * (`creative_director`, `copywriter`, `brand_strategist`, `visual_director`) generate
     * domain-appropriate outputs offline via dependency-injected `ServiceContainer`.
     *
     * @param brief Non-empty campaign topic brief.
     * @param services Optional injected `ServiceContainer` for offline LLM execution.
     * @param config Optional `PitchConfig` override.
     * @return Dictionary mapping each specialist's `output_key` to its generated text.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string")

    clean_brief = brief.strip()
    active_services = services or get_default_services(config)
    team = build_specialist_team(active_services.config)
    model_id = active_services.config.flash_model

    concept_out = active_services.llm.generate_text(
        clean_brief,
        system_instruction=team["creative_director"].instruction,
        model=model_id,
    )
    copy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["copywriter"].instruction,
        model=model_id,
    )
    copy_words = copy_out.split()
    if len(copy_words) > 25:
        copy_out = " ".join(copy_words[:24])

    strategy_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["brand_strategist"].instruction,
        model=model_id,
    )
    visual_out = active_services.llm.generate_text(
        concept_out,
        system_instruction=team["visual_director"].instruction,
        model=model_id,
    )

    return {
        "creative_director": concept_out,
        "copywriter": copy_out,
        "brand_strategist": strategy_out,
        "visual_director": visual_out,
    }


execute_specialists = run_specialist_team
run_specialists = run_specialist_team

# [Guidepost — Step 1a: Author and Attach the Brand Guidelines Skill]
# TODO (Step 1a): Create the house brand skill and attach it to `visual_director`:
#   1. Create `pitch_generator/skills/brand-guidelines/SKILL.md` with YAML frontmatter
#      (`name: brand-guidelines`, `description: ... Use when ...`) and five Markdown sections:
#      `## Palette` (deep indigo, warm amber, slate, terracotta),
#      `## Light` (warm raking golden-hour light with long soft shadows),
#      `## Composition` (off-center hero subject, generous negative space, shallow depth of field),
#      `## Subject` (one realistic photographic subject, authentic/handcrafted materials), and
#      `## Never` (no neon, fluorescent cyan/magenta, watermarks, logos, ring light, or 3D renders).
#      Keep these 5 sections, but set values to match learner's prompt incase they get creative.
#   2. Define `Skill(name, description, body, path=None)` and `SkillToolset(skills)` (with
#      `.load_skill(skill_name) -> str` and `.loaded_skills: list[str]`), plus
#      `load_skill_from_dir(skill_dir: str | Path) -> Skill` and `load_brand_skill() -> Skill`.
#   3. Load `brand_guidelines_skill = load_skill_from_dir(...)` from
#      `pitch_generator/skills/brand-guidelines`, attach
#      `tools=[SkillToolset([brand_guidelines_skill]), generate_key_visual]` to `visual_director`,
#      and update `visual_director.instruction` to first call `load_skill("brand-guidelines")`,
#      write the art direction, call `generate_key_visual`, and output ONLY the art direction text.

# [Guidepost — Step 1b: Skill Evaluation Harness]
# TODO (Step 1b): Build a deterministic evaluation harness to verify skill activation and brand compliance:
#   - `FORBIDDEN_BRAND_PATTERNS: dict[str, str]`: regex patterns for forbidden visual styles
#     (`neon`, `fluorescent`, `cyan`, `magenta`, `watermark`, `logo`, `3d_render`, `ring_light`).
#   - `SkillEvalResult` dataclass with fields: `score: float`, `passed: bool`, `loaded_skill: bool`,
#     `violations: list[str]`, and `pillar_scores: dict[str, float]`.
#   - `evaluate_brand_skill(art_direction: str, loaded_skills: Sequence[str] = ("brand-guidelines",)) -> SkillEvalResult`:
#     verifies `"brand-guidelines"` is in `loaded_skills`, scores the 4 positive pillars
#     (`palette`, `light`, `composition`, `subject` at `0.25` each), deducts `0.25` per forbidden match,
#     and sets `passed = bool(loaded_skill and not violations and score >= 0.75)`.
#   - `run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]`:
#     runs compliant, off-brand, and unloaded-skill benchmark cases and returns a summary dict with
#     `"total"`, `"passed"`, `"results"`, and `"all_expectations_met"`.

FORBIDDEN_BRAND_PATTERNS: dict[str, str] = {
    "neon": r"\bneon\b",
    "fluorescent": r"\bfluorescent\b",
    "cyan": r"\bcyan\b",
    "magenta": r"\bmagenta\b",
    "watermark": r"\bwatermark(?:s)?\b",
    "logo": r"\blogo(?:s)?\b",
    "3d_render": r"\b(?:3d\s+render(?:s)?|cgi)\b",
    "ring_light": r"\bring\s+light\b",
}

POSITIVE_PILLAR_PATTERNS: dict[str, list[str]] = {
    "palette": [r"\bindigo\b", r"\bslate\b", r"\bamber\b", r"\bterracotta\b"],
    "light": [r"\braking\b", r"\bgolden[- ]hour\b", r"\blong\s+(?:soft\s+)?shadows\b"],
    "composition": [r"\boff[- ]center\b", r"\bnegative\s+space\b", r"\bshallow\s+depth\b"],
    "subject": [r"\brealistic\b", r"\bphotographic\b", r"\bauthentic\b", r"\bhandcrafted\b"],
}


@dataclass
class SkillEvalResult:
    """
    /**
     * Evaluation result for brand guidelines compliance and skill activation.
     *
     * @param score Overall compliance score between 0.0 and 1.0.
     * @param passed Whether the art direction passed verification (score >= 0.75, loaded, no violations).
     * @param loaded_skill Whether the required 'brand-guidelines' skill was loaded.
     * @param violations List of forbidden style violations detected.
     * @param pillar_scores Dictionary of scores across the four positive pillars (palette, light, composition, subject).
     */
    """
    score: float
    passed: bool
    loaded_skill: bool
    violations: list[str]
    pillar_scores: dict[str, float]


def evaluate_brand_skill(
    art_direction: str,
    loaded_skills: Sequence[str] = ("brand-guidelines",),
) -> SkillEvalResult:
    """
    /**
     * Deterministically evaluate art direction against house brand guidelines.
     *
     * @param art_direction Generated art direction text.
     * @param loaded_skills List or tuple of skills that were loaded by the agent.
     * @return SkillEvalResult with score, pass/fail status, and breakdown.
     */
    """
    is_loaded = "brand-guidelines" in loaded_skills
    text_lower = (art_direction or "").lower()

    # Detect forbidden violations
    violations: list[str] = []
    for violation_name, pattern in FORBIDDEN_BRAND_PATTERNS.items():
        if re.search(pattern, text_lower, re.IGNORECASE):
            violations.append(violation_name)

    # Score positive pillars (0.25 each)
    pillar_scores: dict[str, float] = {}
    for pillar, patterns in POSITIVE_PILLAR_PATTERNS.items():
        matched = any(re.search(pat, text_lower, re.IGNORECASE) for pat in patterns)
        pillar_scores[pillar] = 0.25 if matched else 0.0

    raw_score = sum(pillar_scores.values()) - (0.25 * len(violations))
    final_score = max(0.0, min(1.0, round(raw_score, 4)))

    passed = bool(is_loaded and not violations and final_score >= 0.75)

    return SkillEvalResult(
        score=final_score,
        passed=passed,
        loaded_skill=is_loaded,
        violations=violations,
        pillar_scores=pillar_scores,
    )


def run_eval_suite(cases: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    /**
     * Run evaluation benchmarks across compliant, off-brand, and unloaded-skill test cases.
     *
     * @param cases Optional list of benchmark test cases.
     * @return Evaluation suite results dictionary.
     */
    """
    if cases is None:
        cases = [
            {
                "name": "fully_compliant_case",
                "art_direction": (
                    "A realistic photographic subject captured in deep indigo and slate with warm amber accents. "
                    "Warm raking golden-hour light casts long soft shadows. Framed with an off-center hero subject "
                    "surrounded by generous negative space and a shallow depth of field, highlighting handcrafted textures."
                ),
                "loaded_skills": ["brand-guidelines"],
                "expected_pass": True,
            },
            {
                "name": "off_brand_neon_case",
                "art_direction": (
                    "A cat hovering with bright neon glow, glowing cyan lasers, and a 3D render metallic logo."
                ),
                "loaded_skills": ["brand-guidelines"],
                "expected_pass": False,
            },
            {
                "name": "missing_skill_case",
                "art_direction": (
                    "A realistic photographic portrait in deep indigo tones with raking golden-hour light, "
                    "off-center framing, and handcrafted wood textures."
                ),
                "loaded_skills": [],
                "expected_pass": False,
            },
        ]

    results: list[dict[str, Any]] = []
    all_expectations_met = True
    passed_count = 0

    for c in cases:
        eval_res = evaluate_brand_skill(
            c.get("art_direction", ""),
            c.get("loaded_skills", ("brand-guidelines",)),
        )
        expected = c.get("expected_pass")
        matches_expectation = expected is None or (eval_res.passed == expected)
        if not matches_expectation:
            all_expectations_met = False

        if eval_res.passed:
            passed_count += 1

        results.append({
            "name": c.get("name", "unnamed_case"),
            "score": eval_res.score,
            "passed": eval_res.passed,
            "loaded_skill": eval_res.loaded_skill,
            "violations": eval_res.violations,
            "pillar_scores": eval_res.pillar_scores,
            "expected_pass": expected,
            "matches_expectation": matches_expectation,
        })

    return {
        "total": len(cases),
        "passed": passed_count,
        "results": results,
        "all_expectations_met": all_expectations_met,
    }

# [Guidepost — Step 1c: Remote A2A Visual Director Service & Client]
# TODO (Step 1c): Expose `visual_director` over the Agent2Agent (A2A) protocol and wire `RemoteA2aAgent`:
#   - `AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"`, `AgentCardBuilder(agent, rpc_url)`,
#     and `build_visual_director_card(rpc_url: str = "http://localhost:8801/a2a/visual_director") -> dict[str, Any]`
#     returning an A2A Agent Card with `"name": "visual_director"`, `"url": rpc_url`, and `"preferredTransport": "JSONRPC"`.
#   - `A2aAgentExecutorConfig`, `include_artifacts_in_a2a_event_interceptor`, and
#     `build_a2a_visual_director_app(services: ServiceContainer | None = None, rpc_url: str = ...) -> Any`
#     configured with `execute_interceptors=[include_artifacts_in_a2a_event_interceptor]` (also exposed in
#     `pitch_generator/fast_api_app.py` when `SERVICE_ROLE == "visual-director"`).
#   - `_cloud_run_client(base_url: str)`: returns `None` for local `http://` or empty URLs, and returns an
#     authenticated HTTP client with `timeout=600.0` and an OIDC `Authorization: Bearer <token>` header for `https://` Cloud Run URLs.
#   - `_pitch_parts_only(part: Any)`: stream filter that drops `function_call` and `function_response` parts
#     (returning `None` / `False`) while keeping `text` and `inline_data` (`image/png`) parts.
#   - `remote_visual_director = RemoteA2aAgent(name="visual_director", description=..., agent_card=f"{visual_director_url}/.well-known/agent-card.json", httpx_client=_cloud_run_client(visual_director_url), genai_part_converter=_pitch_parts_only, output_key="visual_director")`
#     and swap `visual_director` for `remote_visual_director` in `root_agent.edges`. Also remove `hidden`
#     from `#btn-inspect-a2a` in `frontend/index.html` (defining `remote_visual_director` or
#     `build_a2a_visual_director_app` also automatically sets `a2a_enabled=True` in `/api/config`).


def build_visual_director_card(
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> dict[str, Any]:
    """
    /**
     * Build the A2A discovery card for the standalone Visual Director service.
     *
     * @param rpc_url Public URL of the Visual Director JSON-RPC 2.0 endpoint.
     * @return Dictionary containing the A2A v0.3 Agent Card.
     */
    """
    card = dict(AgentCardBuilder(agent=visual_director, rpc_url=rpc_url).build())
    card["name"] = "visual_director"
    card["url"] = rpc_url
    card["preferredTransport"] = "JSONRPC"
    return card


def build_a2a_visual_director_app(
    services: ServiceContainer | None = None,
    rpc_url: str = "http://localhost:8801/a2a/visual_director",
) -> Any:
    """
    /**
     * Factory that creates the standalone Visual Director A2A microservice.
     *
     * @param services Injected ServiceContainer for artifact storage and models.
     * @param rpc_url Base endpoint URL for JSON-RPC 2.0 requests.
     * @return Configured PitchFastAPIApp serving the Visual Director A2A service.
     */
    """
    from pitch_generator.fast_api_app import PitchFastAPIApp

    app_instance = PitchFastAPIApp(services=services)
    executor = A2aAgentExecutor(
        config=A2aAgentExecutorConfig(
            execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
        )
    )
    agent_card = build_visual_director_card(rpc_url=rpc_url)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store={},
        agent_card=agent_card,
    )
    app_instance.register_a2a_handler(
        rpc_path="/a2a/visual_director",
        handler=handler,
        agent_card=agent_card,
    )
    return app_instance


def _cloud_run_client(base_url: str) -> Any:
    """
    /**
     * Build an authenticated HTTP client for Cloud Run A2A endpoints.
     *
     * @param base_url Service target URL.
     * @return httpx.Client with OIDC Bearer token for https:// URLs, else None.
     */
    """
    if not base_url or not base_url.startswith("https://"):
        return None

    import httpx

    headers: dict[str, str] = {}
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        auth_req = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(auth_req, base_url)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    except Exception:
        pass

    return httpx.Client(headers=headers, timeout=600.0)


def _pitch_parts_only(part: Any) -> Any:
    """
    /**
     * Stream filter that drops function_call/response parts while keeping text and png images.
     *
     * @param part GenAI / ADK stream part.
     * @return Part if allowed, else None / False.
     */
    """
    if hasattr(part, "function_call") and getattr(part, "function_call", None) is not None:
        return None
    if hasattr(part, "function_response") and getattr(part, "function_response", None) is not None:
        return None
    if isinstance(part, dict):
        if "function_call" in part or "functionCall" in part:
            return None
        if "function_response" in part or "functionResponse" in part:
            return None
    return part


_vd_url = get_config().visual_director_url.rstrip("/")
remote_visual_director: RemoteA2aAgent = RemoteA2aAgent(
    name="visual_director",
    description=visual_director.description,
    agent_card=f"{_vd_url}{AGENT_CARD_WELL_KNOWN_PATH}",
    httpx_client=_cloud_run_client(_vd_url),
    genai_part_converter=_pitch_parts_only,
    output_key="visual_director",
)

assemble = JoinNode(name="assemble")


def package(
    ctx_or_input: Any,
    node_input: dict[str, Any] | None = None,
) -> _DualSyncAsyncEventStream:
    """
    /**
     * Format and validate the joined specialist outputs into the final campaign pitch.
     *
     * Why: Acts as the terminal node after `assemble = JoinNode(name="assemble")`.
     * Explicitly checks that every expected upstream branch produced non-empty output
     * (raising `ValueError("nothing reached the join from: ...")` if any branch is
     * missing or empty) so silent branch failures are caught immediately. Supports both
     * `package(node_input)` and `package(ctx, node_input)` signatures, strips Markdown
     * code fences from branch values, and attaches `_key_visual(ctx)` when present.
     *
     * @param ctx_or_input Either a `node_input` dict (1-arg form) or `Context` (2-arg form).
     * @param node_input Dictionary of upstream outputs when `ctx_or_input` is a `Context`.
     * @return Dual sync/async stream yielding `Event(content=...)` and `Event(output=...)`.
     */
    """
    if node_input is None:
        if not isinstance(ctx_or_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx: Context | None = None
        actual_input = ctx_or_input
    else:
        if not isinstance(node_input, dict):
            raise ValueError("node_input must be a non-empty dictionary")
        ctx = ctx_or_input if isinstance(ctx_or_input, Context) else None
        actual_input = node_input

    if not actual_input:
        raise ValueError("nothing reached the join from: creative_director, copywriter")

    required_keys = ["creative_director", "copywriter"]
    missing_or_empty = [
        k
        for k in required_keys
        if k not in actual_input or not str(actual_input.get(k) or "").strip()
    ]
    extra_empty = [
        name
        for name, value in actual_input.items()
        if name not in missing_or_empty and not str(value or "").strip()
    ]
    all_empty = missing_or_empty + extra_empty
    if all_empty:
        raise ValueError(f"nothing reached the join from: {', '.join(all_empty)}")

    concept_val = strip_markdown_fences(str(actual_input["creative_director"])).strip()
    copy_val = strip_markdown_fences(str(actual_input["copywriter"])).strip()
    sections = [
        f"CONCEPT\n{concept_val}",
        f"COPY\n{copy_val}",
    ]
    if "brand_strategist" in actual_input and actual_input["brand_strategist"]:
        brand_val = strip_markdown_fences(str(actual_input["brand_strategist"])).strip()
        sections.append(f"BRAND STRATEGY\n{brand_val}")
    if "visual_director" in actual_input and actual_input["visual_director"]:
        art_val = strip_markdown_fences(str(actual_input["visual_director"])).strip()
        sections.append(f"ART DIRECTION\n{art_val}")

    pitch = "\n\n".join(sections)
    parts: list[Any] = [types.Part(text=pitch)]
    image = _key_visual(ctx)
    if image is not None:
        parts.append(types.Part(inline_data=image))

    events = [
        Event(author="package", content=types.Content(role="model", parts=parts)),
        Event(author="package", output=pitch),
    ]
    return _DualSyncAsyncEventStream(events)


loop_guard = LoopGuard(max_iterations=10)

# [Guidepost — Step 3a: PreToolUse Lifecycle Hooks]
DEFAULT_TOOL_ALLOWLIST: frozenset[str] = frozenset({"load_skill", "generate_key_visual"})


class ToolAuthorizationError(PermissionError):
    """
    /**
     * Exception raised when a tool invocation is blocked by PreToolUseHook.
     *
     * Why: Subclasses `PermissionError` so callers can trap authorization denials
     * explicitly while capturing `.tool_name` and `.reason`.
     */
    """

    def __init__(self, message: str, tool_name: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.reason = reason


@dataclass
class HookDecision:
    """
    /**
     * Result of a PreToolUse authorization and argument inspection check.
     */
    """

    allowed: bool
    tool_name: str
    reason: str
    violation_type: str | None = None
    sanitized_args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.allowed

    @property
    def permitted(self) -> bool:
        return self.allowed

    def __bool__(self) -> bool:
        return self.allowed


class PreToolUseHook:
    """
    /**
     * Fail-closed lifecycle hook that validates tool calls before dispatch.
     *
     * Why: Enforces tool allowlists, blocks path traversal and prompt injection markers,
     * rejects oversized payloads, and records every check in an audit trail.
     */
    """

    FORBIDDEN_OVERRIDE_KEYS: frozenset[str] = frozenset({
        "bypass_safety",
        "override_brand_guidelines",
        "disable_safety",
        "skip_validation",
    })

    PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
        "ignore previous instructions",
        "ignore above instructions",
        "disregard previous instructions",
        "<system>",
        "</system>",
        "[system]",
        "[/system]",
        "reveal your system prompt",
    )

    def __init__(
        self,
        allowlist: frozenset[str] | set[str] | Sequence[str] | None = None,
        raise_on_block: bool = False,
    ) -> None:
        if allowlist is None:
            self.allowlist: frozenset[str] = DEFAULT_TOOL_ALLOWLIST
        else:
            self.allowlist = frozenset(allowlist)
        self.raise_on_block = bool(raise_on_block)
        self.audit_log: list[dict[str, Any]] = []

    def validate_tool_call(
        self,
        tool_name: Any,
        tool_args: dict[str, Any] | None = None,
    ) -> HookDecision:
        clean_name = str(tool_name).strip() if tool_name is not None else ""
        args = copy.deepcopy(tool_args) if isinstance(tool_args, dict) else {}

        # 1. Block empty or unlisted tools
        if not clean_name or clean_name not in self.allowlist:
            reason = f"Tool '{clean_name}' is not in the approved PreToolUse allowlist."
            decision = HookDecision(
                allowed=False,
                tool_name=clean_name,
                reason=reason,
                violation_type="unauthorized_tool",
                sanitized_args={},
            )
            return self._record_and_handle(decision)

        # 2. Block forbidden override keys
        for key in self.FORBIDDEN_OVERRIDE_KEYS:
            if key in args:
                reason = f"Forbidden override parameter '{key}' is not permitted."
                decision = HookDecision(
                    allowed=False,
                    tool_name=clean_name,
                    reason=reason,
                    violation_type="forbidden_parameter",
                    sanitized_args={},
                )
                return self._record_and_handle(decision)

        # 3. Inspect arguments for path traversal, oversized strings, and prompt injection
        for arg_key, val in args.items():
            if isinstance(val, str):
                # Oversized string check (> 4096 chars)
                if len(val) > 4096:
                    reason = f"Argument '{arg_key}' exceeds max length (4096 chars, got {len(val)})."
                    decision = HookDecision(
                        allowed=False,
                        tool_name=clean_name,
                        reason=reason,
                        violation_type="prompt_injection",
                        sanitized_args={},
                    )
                    return self._record_and_handle(decision)

                # Path traversal check
                if ".." in val or (arg_key in {"filename", "skill_name", "path"} and val.startswith("/")):
                    reason = f"Path traversal detected in argument '{arg_key}': {val!r}."
                    decision = HookDecision(
                        allowed=False,
                        tool_name=clean_name,
                        reason=reason,
                        violation_type="path_traversal",
                        sanitized_args={},
                    )
                    return self._record_and_handle(decision)

                # Prompt injection check
                val_lower = val.lower()
                for pattern in self.PROMPT_INJECTION_PATTERNS:
                    if pattern in val_lower:
                        reason = f"Prompt injection pattern detected in argument '{arg_key}': {pattern!r}."
                        decision = HookDecision(
                            allowed=False,
                            tool_name=clean_name,
                            reason=reason,
                            violation_type="prompt_injection",
                            sanitized_args={},
                        )
                        return self._record_and_handle(decision)

        # Authorized call
        decision = HookDecision(
            allowed=True,
            tool_name=clean_name,
            reason="Tool invocation authorized.",
            violation_type=None,
            sanitized_args=args,
        )
        return self._record_and_handle(decision)

    def _record_and_handle(self, decision: HookDecision) -> HookDecision:
        audit_entry = {
            "tool_name": decision.tool_name,
            "allowed": decision.allowed,
            "reason": decision.reason,
            "violation_type": decision.violation_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.audit_log.append(audit_entry)
        if not decision.allowed and self.raise_on_block:
            raise ToolAuthorizationError(
                decision.reason,
                tool_name=decision.tool_name,
                reason=decision.reason,
            )
        return decision


_DEFAULT_PRE_TOOL_HOOK = PreToolUseHook()


def validate_tool_call(
    tool_name: Any,
    tool_args: dict[str, Any] | None = None,
    hook: PreToolUseHook | None = None,
) -> HookDecision:
    """
    /**
     * Module-level helper validating a tool call via PreToolUseHook.
     */
    """
    active_hook = hook or _DEFAULT_PRE_TOOL_HOOK
    return active_hook.validate_tool_call(tool_name, tool_args)

# [Guidepost — Step 3c: Human-in-the-Loop (HITL) Approval Gate]
@node(rerun_on_resume=False)
async def approve_concept(ctx: Context) -> AsyncIterator[RequestInput]:
    """
    /**
     * Pauses the workflow after Creative Director to request human approval.
     *
     * Why: Enforces fail-closed human authorization before executing downstream copy
     * and image generation specialist agents. Marked `rerun_on_resume=False` so resumed
     * sessions read cached reviewer responses rather than pausing again.
     *
     * @param ctx Execution context containing session state.
     * @return Async generator yielding `RequestInput`.
     */
    """
    concept = ctx.session.state.get("creative_director")
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("Cannot approve empty concept draft from creative_director")
    yield RequestInput(
        message="Please approve the campaign concept (yes/no).",
        response_schema=str,
    )


def evaluate_user_approval(concept: str, decision: str) -> str:
    """
    /**
     * Evaluates a human reviewer's approval decision.
     *
     * Why: Accepts affirmative responses ("yes", "y", "true", "approved") and emits
     * formatted markdown with the `## Approved Concept` header. Rejects all other
     * inputs fail-closed with `ValueError("User rejected the concept")`.
     *
     * @param concept Campaign concept text.
     * @param decision User decision string.
     * @return Formatted approved concept markdown.
     */
    """
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("Concept must not be empty")
    clean_dec = str(decision).strip().lower() if decision is not None else ""
    affirmative = {"yes", "y", "true", "approved"}
    if clean_dec in affirmative:
        return f"## Approved Concept\n\n{concept.strip()}"
    raise ValueError("User rejected the concept")


@node(rerun_on_resume=True)
async def user_approval(
    ctx: Context,
    node_input: dict[str, Any] | None = None,
) -> AsyncIterator[Event]:
    """
    /**
     * Resumes the paused workflow, verifies human approval, and emits the approved concept.
     *
     * Why: Marked `rerun_on_resume=True` so validation logic re-executes on resume,
     * confirming the human reviewer's affirmative authorization before downstream dispatch.
     *
     * @param ctx Execution context with cached reviewer input.
     * @param node_input Optional upstream node inputs.
     * @return Async generator yielding an `Event` with author="user_approval".
     */
    """
    concept = ctx.session.state.get("creative_director")
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("No creative_director concept available to approve")

    approval_input = await ctx.run_node(approve_concept)
    decision_str = getattr(approval_input, "response", approval_input)
    if isinstance(decision_str, RequestInput):
        decision_str = ctx.session.state.get("approval_response", "")

    approved_content = evaluate_user_approval(concept, str(decision_str))
    yield Event(
        author="user_approval",
        content=approved_content,
        output={"approved_concept": approved_content},
    )


def run_hitl_workflow(
    brief: str,
    *,
    approved: bool | None = None,
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute or resume a Human-in-the-Loop gated campaign pitch workflow.
     *
     * Why: Provides a testable entrypoint returning `status="input_required"` (0 downstream calls)
     * when `approved is None`, `status="rejected"` (0 downstream calls) when `approved is False`,
     * and `status="completed"` (2 downstream calls) when `approved is True`.
     *
     * @param brief Campaign topic brief.
     * @param approved Approval flag (`None`=pause, `False`=reject, `True`=approve).
     * @param services Injected service container.
     * @return Workflow execution result dictionary.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must not be empty")

    clean_brief = brief.strip()
    active_services = services or get_default_services()
    active_model = active_services.config.flash_model

    # Run Creative Director
    concept = active_services.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=active_model,
    )

    if approved is None:
        return {
            "status": "input_required",
            "prompt": "Please approve the campaign concept (yes/no).",
            "concept": concept,
            "copy": "",
            "art_direction": "",
            "downstream_calls": 0,
        }

    if approved is False:
        return {
            "status": "rejected",
            "concept": concept,
            "copy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "downstream_calls": 0,
        }

    # Approved: Evaluate and continue with downstream specialists (copywriter + visual_director)
    concept_header = evaluate_user_approval(concept, "yes")
    copy_text = active_services.llm.generate_text(
        concept_header,
        system_instruction=copywriter.instruction,
        model=active_model,
    )
    effective_visual_instruction = (
        f"{visual_director.instruction}\n\n"
        "Return ONLY the written art direction notes as plain text "
        "(do not emit tool call syntax; write the art direction prompt directly)."
    )
    art_dir = active_services.llm.generate_text(
        concept_header,
        system_instruction=effective_visual_instruction,
        model=active_model,
    )
    return {
        "status": "completed",
        "concept": concept_header,
        "copy": copy_text,
        "art_direction": art_dir,
        "downstream_calls": 2,
    }


root_agent = Workflow(
    name="pitch_generator",
    edges=[
        (creative_director, user_approval, brand_strategist, (copywriter, remote_visual_director)),
        ((copywriter, remote_visual_director), assemble, package),
    ],
)
loop_guard.validate_graph(root_agent.edges)

app = App(root_agent=root_agent, name="pitch_generator")


# [Guidepost — Step 4a: Tokenomics & Context Optimization]
def estimate_tokens(text: str | None) -> int:
    """
    /**
     * Estimates token count deterministically using the standard 4-chars-per-token heuristic.
     *
     * Why: Provides lightweight, offline token estimation for conversation history turns,
     * system instructions, and prompt caching calculations without invoking a tokenizer.
     *
     * @param text Input text string or None.
     * @return Estimated token count (0 for empty/whitespace, max(1, len // 4) otherwise).
     */
    """
    if text is None:
        return 0
    clean = text.strip()
    if not clean:
        return 0
    return max(1, len(clean) // 4)


class CompressedHistoryList(list):
    """
    /**
     * List subclass holding optimized conversation history turns along with token metrics.
     *
     * Why: Preserves standard list behaviors for graph runners and serialization while
     * carrying tokenomics metadata (.turns, .history, .summary, .original_tokens,
     * .optimized_tokens, .tokens_saved).
     */
    """

    def __init__(
        self,
        items: Sequence[dict[str, Any]] | None = None,
        *,
        summary: str = "",
        original_tokens: int = 0,
        optimized_tokens: int = 0,
    ) -> None:
        super().__init__(items or [])
        self.summary: str = summary
        self.original_tokens: int = int(original_tokens)
        self.optimized_tokens: int = int(optimized_tokens)

    @property
    def turns(self) -> list[dict[str, Any]]:
        return list(self)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.optimized_tokens)


def compress_memory(
    turns: Sequence[dict[str, Any]],
    keep_recent: int = 2,
) -> CompressedHistoryList:
    """
    /**
     * Compresses older conversation turns into a synthetic system summary.
     *
     * Why: Reduces context window token consumption on multi-turn pitch refinement
     * sessions by condensing older discussion while preserving the last `keep_recent`
     * turns verbatim.
     *
     * @param turns Sequence of conversation turn dictionaries.
     * @param keep_recent Number of recent turns to preserve verbatim.
     * @return CompressedHistoryList with summary and recent turns.
     */
    """
    if not turns:
        return CompressedHistoryList([], summary="", original_tokens=0, optimized_tokens=0)

    turn_list = [copy.deepcopy(t) for t in turns]
    orig_tokens = sum(estimate_tokens(str(t.get("content", ""))) for t in turn_list)

    if len(turn_list) <= keep_recent:
        return CompressedHistoryList(
            turn_list,
            summary="",
            original_tokens=orig_tokens,
            optimized_tokens=orig_tokens,
        )

    older_turns = turn_list[:-keep_recent]
    recent_turns = turn_list[-keep_recent:]

    summary_snippets = []
    for t in older_turns:
        role = t.get("role", "user")
        cnt = str(t.get("content", "")).strip()
        summary_snippets.append(f"{role}: {cnt[:60]}...")
    summary_text = "Summary of prior discussion: " + " | ".join(summary_snippets)

    system_turn = {
        "role": "system",
        "compressed": True,
        "content": summary_text,
    }
    result_turns = [system_turn] + recent_turns
    opt_tokens = sum(estimate_tokens(str(t.get("content", ""))) for t in result_turns)

    return CompressedHistoryList(
        result_turns,
        summary=summary_text,
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
    )


def prune_history(
    turns: Sequence[dict[str, Any]],
    max_turns: int = 6,
    max_tokens: int = 512,
) -> CompressedHistoryList:
    """
    /**
     * Prunes conversation history using a sliding-window constrained by turn and token budgets.
     *
     * Why: Guarantees a hard upper bound on conversation history tokens sent to the model,
     * keeping only the newest turns and truncating a single oversized turn to fit budget.
     *
     * @param turns Sequence of conversation turns.
     * @param max_turns Maximum number of turns to keep.
     * @param max_tokens Maximum total estimated token budget.
     * @return CompressedHistoryList meeting both turn and token constraints.
     */
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    if not turns:
        return CompressedHistoryList([], summary="", original_tokens=0, optimized_tokens=0)

    turn_list = [copy.deepcopy(t) for t in turns]
    orig_tokens = sum(estimate_tokens(str(t.get("content", ""))) for t in turn_list)

    # First window by max_turns (keep newest turns)
    window = turn_list[-max_turns:]

    # If single turn is oversized and exceeds max_tokens, truncate its content
    if len(window) == 1:
        t = window[0]
        tok = estimate_tokens(str(t.get("content", "")))
        if tok > max_tokens:
            max_chars = max(1, max_tokens * 4)
            raw_content = str(t.get("content", ""))
            truncated_turn = copy.deepcopy(t)
            truncated_turn["content"] = raw_content[:max_chars]
            truncated_turn["truncated"] = True
            opt_tokens = estimate_tokens(truncated_turn["content"])
            return CompressedHistoryList(
                [truncated_turn],
                summary="",
                original_tokens=orig_tokens,
                optimized_tokens=opt_tokens,
            )
        return CompressedHistoryList(
            window,
            summary="",
            original_tokens=orig_tokens,
            optimized_tokens=tok,
        )

    # For multiple turns, keep newest turns from right to left that fit within max_tokens
    selected: list[dict[str, Any]] = []
    accum_tokens = 0
    for t in reversed(window):
        t_tokens = estimate_tokens(str(t.get("content", "")))
        if accum_tokens + t_tokens <= max_tokens:
            selected.append(t)
            accum_tokens += t_tokens
        else:
            break

    selected.reverse()
    if not selected and window:
        # Fallback: truncate the single latest turn to fit
        latest = window[-1]
        max_chars = max(1, max_tokens * 4)
        raw_content = str(latest.get("content", ""))
        trunc = copy.deepcopy(latest)
        trunc["content"] = raw_content[:max_chars]
        trunc["truncated"] = True
        selected = [trunc]
        accum_tokens = estimate_tokens(trunc["content"])

    return CompressedHistoryList(
        selected,
        summary="",
        original_tokens=orig_tokens,
        optimized_tokens=accum_tokens,
    )


class PromptCacheManager:
    """
    /**
     * Manages SHA-256 prompt caching with TTL expiration for static prompt prefixes.
     *
     * Why: Avoids recomputing and resending invariant system instructions and house
     * brand guidelines across multi-turn sessions, tracking cache hits and saved tokens.
     */
    """

    def __init__(self, default_ttl_seconds: int = 300) -> None:
        self.default_ttl_seconds: int = default_ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}

    def get_or_create(
        self,
        static_prompt: str,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """
        /**
         * Look up or register a cached static prompt prefix by SHA-256 hash.
         *
         * Why: Tracks cache hit/miss status and calculates saved tokens.
         *
         * @param static_prompt Non-empty prompt string to cache.
         * @param ttl_seconds Optional TTL override in seconds.
         * @return Cache metadata dictionary.
         */
        """
        if not isinstance(static_prompt, str) or not static_prompt.strip():
            raise ValueError("static_prompt must be a non-empty string")

        clean_text = static_prompt.strip()
        cache_key = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        now = time.time()

        entry = self._cache.get(cache_key)
        tokens = estimate_tokens(clean_text)

        if entry is not None and entry["expires_at"] > now and ttl > 0:
            entry["hit_count"] += 1
            return {
                "cache_key": cache_key,
                "cache_hit": True,
                "saved_tokens": tokens,
                "tokens": tokens,
                "ttl_seconds": ttl,
            }

        # Cache miss or expired
        expires_at = now + ttl
        self._cache[cache_key] = {
            "cache_key": cache_key,
            "text": clean_text,
            "tokens": tokens,
            "created_at": now,
            "expires_at": expires_at,
            "hit_count": 0,
        }
        return {
            "cache_key": cache_key,
            "cache_hit": False,
            "saved_tokens": 0,
            "tokens": tokens,
            "ttl_seconds": ttl,
        }


def select_model_strategy(tier: str = "flash") -> dict[str, Any]:
    """
    /**
     * Maps a model strategy tier to its model ID, target hardware/cloud endpoint,
     * cost tier, and thinking budget.
     *
     * Why: Provides a standardized 5-tier routing taxonomy (flash, pro, low, high, local)
     * balancing latency, cost, and reasoning depth across agent tasks.
     *
     * @param tier Strategy tier identifier.
     * @return Dictionary with strategy configuration metadata.
     */
    """
    clean_tier = str(tier).strip().lower()
    strategies: dict[str, dict[str, Any]] = {
        "flash": {
            "tier": "flash",
            "model_id": "gemini-3.8-flash",
            "target": "cloud",
            "cost_tier": "low",
            "thinking_budget": 0,
        },
        "pro": {
            "tier": "pro",
            "model_id": "gemini-3.8-pro",
            "target": "cloud",
            "cost_tier": "high",
            "thinking_budget": 2048,
        },
        "low": {
            "tier": "low",
            "model_id": "gemini-3.8-flash",
            "target": "cloud",
            "cost_tier": "low",
            "thinking_budget": 0,
        },
        "high": {
            "tier": "high",
            "model_id": "gemini-3.8-pro",
            "target": "cloud",
            "cost_tier": "high",
            "thinking_budget": 4096,
        },
        "local": {
            "tier": "local",
            "model_id": "gemma-2-2b-local",
            "target": "edge",
            "cost_tier": "zero",
            "thinking_budget": 0,
        },
    }

    if clean_tier not in strategies:
        raise ValueError(
            f"Unsupported model strategy tier: {tier!r}. "
            f"Must be one of: {list(strategies.keys())}"
        )

    return copy.deepcopy(strategies[clean_tier])


class TokenomicsManager:
    """
    /**
     * Unified manager orchestrating memory compression, sliding-window history pruning,
     * SHA-256 prompt caching, and model tier selection.
     *
     * Why: Ties together all tokenomics optimization pillars to maximize context efficiency
     * and minimize token costs across agent workflow executions.
     */
    """

    def __init__(
        self,
        max_turns: int = 6,
        max_tokens: int = 512,
        cache_manager: PromptCacheManager | None = None,
    ) -> None:
        self.max_turns: int = max_turns
        self.max_tokens: int = max_tokens
        self.cache_manager: PromptCacheManager = cache_manager or PromptCacheManager()

    def optimize(
        self,
        turns: Sequence[dict[str, Any]],
        static_prompt: str = "",
        max_turns: int | None = None,
        max_tokens: int | None = None,
        tier: str = "flash",
    ) -> dict[str, Any]:
        """
        /**
         * Optimizes turns, static prompt prefix, and model strategy in a single pass.
         *
         * Why: Quantifies end-to-end token and cost savings for campaign pitch sessions.
         *
         * @param turns History turns list.
         * @param static_prompt Optional invariant system prompt or brand guidelines.
         * @param max_turns Optional override for maximum history turns.
         * @param max_tokens Optional override for token budget.
         * @param tier Model strategy tier.
         * @return Optimization summary dictionary.
         */
        """
        effective_max_turns = self.max_turns if max_turns is None else max_turns
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens

        strategy = select_model_strategy(tier)
        orig_tokens = sum(estimate_tokens(str(t.get("content", ""))) for t in turns)

        # 1. Memory compression
        compressed = compress_memory(turns, keep_recent=2)

        # 2. History pruning to hard limits
        pruned = prune_history(
            compressed,
            max_turns=effective_max_turns,
            max_tokens=effective_max_tokens,
        )

        # 3. Prompt caching
        cache_result = None
        cache_hit = False
        cache_key = ""
        saved_cache_tokens = 0
        if static_prompt and static_prompt.strip():
            cache_result = self.cache_manager.get_or_create(static_prompt)
            cache_hit = cache_result["cache_hit"]
            cache_key = cache_result["cache_key"]
            saved_cache_tokens = cache_result.get("saved_tokens", 0)

        optimized_tokens = pruned.optimized_tokens
        tokens_saved = max(0, orig_tokens - optimized_tokens) + saved_cache_tokens

        return {
            "turns": pruned.turns,
            "original_tokens": orig_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "cache_result": cache_result,
            "selected_strategy": strategy,
        }


# ==============================================================================
# Internal Workflow Execution Engine (Do not modify unless extending runners)
# ==============================================================================


def run_graph_workflow(
    brief: str,
    *,
    services: ServiceContainer | None = None,
    max_iterations: int = 10,
) -> PitchPackage:
    """
    /**
     * Validates the workflow DAG with `LoopGuard` and executes the fan-out/fan-in pitch graph.
     *
     * Why: Provides a deterministic, offline-safe execution helper that validates the
     * graph topology, tracks every node visit in `LoopGuard`, parses structured payloads,
     * and returns a complete `PitchPackage`.
     *
     * @param brief Campaign product brief string.
     * @param services Optional `ServiceContainer` for offline LLM/artifact services.
     * @param max_iterations Maximum allowed node visits in `LoopGuard`.
     * @return Populated `PitchPackage` instance.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must be a non-empty string.")

    guard = LoopGuard(max_iterations=max_iterations)
    guard.validate_graph(root_agent.edges)

    svc = services or get_default_services()
    ctx = Context(services=svc)
    clean_brief = brief.strip()
    model_id = svc.config.flash_model

    guard.record_step("creative_director")
    concept_raw = svc.llm.generate_text(
        clean_brief,
        system_instruction=creative_director.instruction,
        model=model_id,
    )
    concept_lines = [ln.strip() for ln in concept_raw.splitlines() if ln.strip()]
    concept = ConceptPayload(
        concept_line=concept_lines[0] if concept_lines else f"Engineered for everyday motion: {clean_brief}.",
        rationale=(
            concept_lines[1]
            if len(concept_lines) > 1
            else "Grounded in practical commuter utility and understated design."
        ),
    )

    guard.record_step("brand_strategist")
    strategy_notes = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=brand_strategist.instruction,
        model=model_id,
    )

    guard.record_step("copywriter")
    copy_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=copywriter.instruction,
        model=model_id,
    )
    copy_payload = CopyPayload(caption=copy_text)

    guard.record_step("visual_director")
    art_text = svc.llm.generate_text(
        concept.concept_line,
        system_instruction=visual_director.instruction,
        model=model_id,
    )
    art_payload = ArtDirectionPayload(art_direction=art_text)

    guard.record_step("assemble")
    joined_inputs = {
        "creative_director": f"{concept.concept_line}\n{concept.rationale}",
        "brand_strategist": strategy_notes,
        "copywriter": copy_payload.caption,
        "visual_director": art_payload.art_direction,
    }

    guard.record_step("package")
    packaged_events = list(package(ctx, joined_inputs))
    final_text = ""
    for ev in packaged_events:
        if ev.output:
            final_text = str(ev.output)

    return PitchPackage(
        concept=concept,
        copy=copy_payload,
        art_direction=art_payload,
        brand_strategy=strategy_notes,
        pitch_text=final_text,
        trace=list(guard.visited_nodes),
    )


execute_workflow = run_graph_workflow
run_workflow = run_graph_workflow


def run_pitch_workflow(
    brief: str,
    *,
    session_id: str = "default",
    services: ServiceContainer | None = None,
    approved: bool | None = True,
    require_approval: bool = False,
    existing_concept: str | None = None,
) -> dict[str, Any]:
    """
    /**
     * Execute the Pitch Generator multi-agent workflow using injected cloud services.
     *
     * Why: Coordinates the 4-specialist pipeline (`creative_director` -> `brand_strategist`
     * -> (`copywriter`, `visual_director`) -> `assemble` -> `package`) on Gemini Enterprise
     * Agent Platform (`gemini-3.8-flash`), persists session state in `MemoryBankService`,
     * and records telemetry via `BigQueryAnalyticsService`. When the learner adds the HITL
     * approval gate (`approve_concept` / `user_approval`) in Module 3, this runner dynamically
     * pauses and resumes at the concept review gate.
     *
     * @param brief Non-empty campaign topic brief.
     * @param session_id Session identifier for Memory Bank persistence.
     * @param services Optional injected `ServiceContainer` (defaults to `get_default_services()`).
     * @param approved HITL approval state (`True` to complete, `False` to reject,
     *   `None` to pause at `input_required` when HITL is enabled).
     * @param require_approval If True, pauses at the HITL gate (`status="input_required"`).
     * @param existing_concept Optional pre-generated concept to reuse when resuming a paused
     *   HITL session so `creative_director` is not re-executed.
     * @return Workflow result dictionary with keys `session_id`, `status`, `concept`,
     *   `copy`, `brand_strategy`, `art_direction`, `key_visual_uri`, `telemetry`,
     *   `artifacts`, `pitch_text`, and `trace`.
     */
    """
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Campaign brief must not be empty")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")

    clean_brief = brief.strip()
    clean_session = session_id.strip()

    # Step 3b Defense-in-depth: Scrub PII from brief before model or log calls
    try:
        from pitch_generator.app_utils.services import scrub_text
        clean_brief = str(scrub_text(clean_brief).scrubbed).strip()
    except Exception:
        pass

    active_services = services or get_default_services()
    module_globals = globals()
    strip_fn = module_globals.get("strip_markdown_fences")
    active_model = active_services.config.flash_model

    # Step 1: Creative Director generates campaign concept & rationale (or reuses saved concept on resume)
    if isinstance(existing_concept, str) and existing_concept.strip():
        concept = existing_concept.strip()
    else:
        concept = active_services.llm.generate_text(
            clean_brief,
            system_instruction=creative_director.instruction,
            model=active_model,
        )
        if callable(strip_fn):
            concept = strip_fn(concept)

    # Determine effective approval state (used when learner enables HITL in Module 3)
    effective_approved: bool | None = approved
    if require_approval and approved is True:
        effective_approved = None

    if effective_approved is False:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "rejected",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept", "user_approval"],
            }
        )
        rejected_state = {
            "session_id": clean_session,
            "status": "rejected",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept", "user_approval"],
        }
        active_services.memory_bank.save_session(clean_session, rejected_state)
        return rejected_state

    if effective_approved is None:
        telemetry_event = active_services.analytics.record_telemetry(
            {
                "session_id": clean_session,
                "status": "input_required",
                "brief": clean_brief,
                "model_id": active_model,
                "nodes_executed": ["creative_director", "approve_concept"],
            }
        )
        paused_state = {
            "session_id": clean_session,
            "status": "input_required",
            "prompt": "Please approve the campaign concept (yes/no).",
            "brief": clean_brief,
            "concept": concept,
            "copy": "",
            "brand_strategy": "",
            "art_direction": "",
            "key_visual_uri": None,
            "telemetry": telemetry_event,
            "artifacts": [],
            "pitch_text": "",
            "trace": ["creative_director", "approve_concept"],
        }
        active_services.memory_bank.save_session(clean_session, paused_state)
        return paused_state

    # Step 2: Copywriter generates <25-word social caption
    copy_text = active_services.llm.generate_text(
        concept,
        system_instruction=copywriter.instruction,
        model=active_model,
    )

    hitl_defined = (
        module_globals.get("approve_concept") is not None
        or module_globals.get("user_approval") is not None
    )
    join_inputs: dict[str, Any] = {
        "creative_director": concept,
        "copywriter": copy_text,
    }
    nodes_executed: list[str] = (
        ["creative_director", "approve_concept", "user_approval", "copywriter"]
        if hitl_defined
        else ["creative_director", "copywriter"]
    )
    brand_text = ""
    art_direction = ""
    key_visual_uri: str | None = None
    key_visual_url: str | None = None
    artifacts_meta: list[dict[str, Any]] = []

    loop_guard_cls = module_globals.get("LoopGuard")
    active_guard = None
    if callable(loop_guard_cls):
        active_guard = loop_guard_cls(max_iterations=10)
        if hasattr(active_guard, "validate_graph") and hasattr(root_agent, "edges"):
            active_guard.validate_graph(root_agent.edges)
        if hasattr(active_guard, "record_step"):
            active_guard.record_step("creative_director")
            active_guard.record_step("copywriter")

    if callable(strip_fn):
        copy_text = strip_fn(copy_text)
        join_inputs["creative_director"] = concept
        join_inputs["copywriter"] = copy_text

    brand_agent = module_globals.get("brand_strategist")
    if brand_agent is None:
        brand_agent = next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "brand_strategist"
            ),
            None,
        )
    if brand_agent is not None:
        brand_instruction = getattr(
            brand_agent,
            "instruction",
            "You are the Brand Strategist. Provide audience positioning for the concept.",
        )
        brand_text = active_services.llm.generate_text(
            concept,
            system_instruction=brand_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            brand_text = strip_fn(brand_text)
        join_inputs["brand_strategist"] = brand_text
        nodes_executed.append("brand_strategist")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("brand_strategist")

    remote_vd = module_globals.get("remote_visual_director")
    if remote_vd is None:
        remote_vd = next(
            (v for v in module_globals.values() if isinstance(v, RemoteA2aAgent)),
            None,
        )
    visual_agent = (
        module_globals.get("visual_director")
        or next(
            (
                v
                for v in module_globals.values()
                if isinstance(v, Agent) and getattr(v, "name", "") == "visual_director"
            ),
            None,
        )
        or remote_vd
    )
    if visual_agent is not None:
        if remote_vd is not None:
            from pitch_generator.app_utils.services import _is_offline_test_mode
            import urllib.request

            if not _is_offline_test_mode(active_services.config.project_id):
                vd_url = active_services.config.visual_director_url.rstrip("/")
                card_url = f"{vd_url}/.well-known/agent-card.json"
                headers: dict[str, str] = {"Accept": "application/json"}
                if card_url.startswith("https://"):
                    try:
                        import google.auth.transport.requests
                        import google.oauth2.id_token

                        auth_req = google.auth.transport.requests.Request()
                        id_tok = google.oauth2.id_token.fetch_id_token(auth_req, vd_url)
                        if id_tok:
                            headers["Authorization"] = f"Bearer {id_tok}"
                    except Exception:
                        pass
                try:
                    req = urllib.request.Request(card_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}")
                except Exception as exc:
                    raise RuntimeError(
                        f"[A2A Remote Error] Could not connect to remote Visual Director service at "
                        f"'{vd_url}' ({card_url}): {exc}. Ensure the Visual Director service is running "
                        f"(or verify VISUAL_DIRECTOR_URL in .env)."
                    ) from exc

        visual_instruction = str(
            getattr(
                visual_agent,
                "instruction",
                (
                    "You are the Visual Director. Produce brand-compliant art direction "
                    "using our deep indigo and slate palette with a warm amber or terracotta "
                    "accent, single low raking light with long shadows, off-center negative "
                    "space, and one realistic photographic subject."
                ),
            )
            or ""
        )
        skill_md_path = Path(__file__).resolve().parent / "skills" / "brand-guidelines" / "SKILL.md"
        if skill_md_path.is_file():
            try:
                skill_text = skill_md_path.read_text(encoding="utf-8").strip()
                if skill_text and skill_text not in visual_instruction:
                    visual_instruction = f"{visual_instruction}\n\n[Loaded Skill: brand-guidelines]\n{skill_text}"
            except OSError:
                pass
        effective_visual_instruction = (
            f"{visual_instruction}\n\n"
            "Return ONLY the written art direction notes as plain text "
            "(do not emit tool call syntax; generate_key_visual will be invoked automatically with your text output)."
        )
        art_direction = active_services.llm.generate_text(
            concept,
            system_instruction=effective_visual_instruction,
            model=active_model,
        )
        if callable(strip_fn):
            art_direction = strip_fn(art_direction)
        image_bytes, mime_type = active_services.llm.generate_image(
            art_direction,
            model=active_services.config.image_model,
        )
        artifact_record = active_services.artifacts.save_artifact(
            filename="key_visual.png",
            data=image_bytes,
            mime_type=mime_type,
            session_id=clean_session,
        )
        key_visual_url = (
            f"/api/artifacts/{clean_session}/{artifact_record.filename}"
            f"?v={artifact_record.version}"
        )
        key_visual_uri = (
            artifact_record.gcs_uri
            or f"/api/artifacts/{clean_session}/{artifact_record.filename}"
        )
        artifacts_meta.append(artifact_record.to_metadata_dict())
        join_inputs["visual_director"] = art_direction
        nodes_executed.append("visual_director")
        if active_guard is not None and hasattr(active_guard, "record_step"):
            active_guard.record_step("visual_director")

    # Step 3: Assemble and Package join output validation
    join_events = list(package(join_inputs))
    pitch_text = str(join_events[-1].output or "")
    nodes_executed.extend(["assemble", "package"])
    if active_guard is not None and hasattr(active_guard, "record_step"):
        active_guard.record_step("assemble")
        active_guard.record_step("package")

    telemetry_event = active_services.analytics.record_telemetry(
        {
            "session_id": clean_session,
            "status": "completed",
            "brief": clean_brief,
            "model_id": active_model,
            "key_visual_uri": key_visual_uri,
            "nodes_executed": list(nodes_executed),
        }
    )

    completed_result = {
        "session_id": clean_session,
        "status": "completed",
        "brief": clean_brief,
        "concept": concept,
        "copy": copy_text,
        "brand_strategy": brand_text,
        "art_direction": art_direction,
        "key_visual_uri": key_visual_uri,
        "key_visual_url": key_visual_url,
        "telemetry": telemetry_event,
        "artifacts": artifacts_meta,
        "pitch_text": pitch_text,
        "trace": list(nodes_executed),
    }

    active_services.memory_bank.save_session(clean_session, completed_result)
    active_services.memory_bank.store_memory(
        "campaigns",
        clean_session,
        {
            "brief": clean_brief,
            "concept": concept,
            "copy": copy_text,
            "key_visual_uri": key_visual_uri,
        },
    )
    return completed_result


def resume_pitch_workflow(
    session_id: str,
    *,
    approved: bool,
    feedback: str = "",
    services: ServiceContainer | None = None,
) -> dict[str, Any]:
    """
    /**
     * Resume a paused Human-in-the-Loop workflow session stored in `MemoryBankService`.
     *
     * Why: When `POST /api/approve` or an A2A approval message arrives for a paused
     * `session_id` (Module 3), this function loads the saved concept from Memory Bank
     * and either completes the downstream workflow (`approved=True`) or marks the
     * session `"rejected"` (`approved=False`).
     *
     * @param session_id Session identifier of the paused workflow.
     * @param approved Boolean decision from the human reviewer.
     * @param feedback Optional reviewer note or refinement instruction.
     * @param services Optional injected `ServiceContainer`.
     * @return Updated workflow result dictionary (`"completed"` or `"rejected"`).
     */
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must not be empty")
    clean_session = session_id.strip()
    active_services = services or get_default_services()
    stored = active_services.memory_bank.load_session(clean_session)
    if not stored:
        raise KeyError(f"Unknown session_id: {clean_session}")

    brief = str(stored.get("brief") or "Campaign brief")
    saved_concept = str(stored.get("concept") or "").strip() or None

    if not approved:
        result = run_pitch_workflow(
            brief,
            session_id=clean_session,
            services=active_services,
            approved=False,
            existing_concept=saved_concept,
        )
        result["reviewer_feedback"] = feedback
        active_services.memory_bank.save_session(clean_session, result)
        return result

    effective_brief = f"{brief} ({feedback.strip()})" if feedback and feedback.strip() else brief
    result = run_pitch_workflow(
        effective_brief,
        session_id=clean_session,
        services=active_services,
        approved=True,
        require_approval=False,
        existing_concept=saved_concept,
    )
    result["reviewer_feedback"] = feedback
    active_services.memory_bank.save_session(clean_session, result)
    return result
