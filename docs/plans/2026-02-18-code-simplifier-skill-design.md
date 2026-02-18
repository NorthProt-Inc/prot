# Code Simplifier Skill Design

## Goal

Create a skill that scans an entire project to detect over-engineered, unnecessarily complex, or AI-agent-generated anti-pattern code, then produces a verified refactoring plan.

## Architecture

Orchestrator (SKILL.md) dispatches 5 perspective-specific worker subagents in parallel, each analyzing the full codebase through a single lens. The orchestrator synthesizes findings into a prioritized refactoring plan, then invokes `/reviewing-plans` for verification before saving.

**Tech stack:** Claude Code skill system (SKILL.md + worker prompt templates)

**Execution:** Use `plan-execution` skill after plan is approved.

## Flow

```
Phase 1: Architecture Snapshot (main agent)
  Read project structure, CLAUDE.md, config, entry points
  Build module dependency map
  Identify project conventions and patterns

Phase 2: Parallel Analysis (5 worker subagents)
  worker-complexity    → cyclomatic complexity, function length, nesting depth
  worker-dry           → duplication, near-duplicates, regeneration-instead-of-reuse
  worker-yagni         → premature abstraction, unused extension points, future-proofing
  worker-dead-code     → unused imports/functions/classes, unreachable paths, refactoring debris
  worker-coupling      → module coupling graph, circular refs, hidden dependencies, god modules

Phase 3: Synthesis (main agent)
  Collect 5 worker reports
  False positive screening (project-qc pattern)
  Cross-perspective deduplication
  Priority scoring: impact × effort × risk
  Draft refactoring plan (writing-plans format)

Phase 4: Verification
  Invoke /reviewing-plans on the draft plan
  Apply reviewer feedback
  Save final plan to docs/plans/YYYY-MM-DD-simplification-plan.md
```

## File Structure

```
~/.claude/skills/code-simplifier/
  SKILL.md                  # Orchestrator: flow control, synthesis, plan writing
  worker-complexity.md      # Subagent prompt: complexity & over-engineering
  worker-dry.md             # Subagent prompt: duplication & DRY violations
  worker-yagni.md           # Subagent prompt: YAGNI & premature abstraction
  worker-dead-code.md       # Subagent prompt: dead code & staleness
  worker-coupling.md        # Subagent prompt: coupling & hidden dependencies
  ai-anti-patterns.md       # Shared reference: AI coding agent mistake catalog
```

## Component Design

### SKILL.md (Orchestrator)

**Responsibilities:**
1. Phase 1: Build architecture snapshot by reading project structure, CLAUDE.md, key config files
2. Phase 2: Dispatch 5 workers in parallel using `dispatching-parallel-agents` pattern
   - Each worker gets: project root path, architecture snapshot, analysis criteria (embedded from existing skills)
3. Phase 3: Synthesize results
   - False positive screening (adapted from project-qc Step 3)
   - Cross-perspective dedup: same code flagged by multiple workers → merge into single finding
   - Priority scoring matrix (see below)
4. Phase 4: Format as writing-plans compatible document, invoke reviewing-plans

**Priority Scoring:**

| Factor | Weight | Scale |
|--------|--------|-------|
| Impact (how much simpler the code becomes) | 40% | 1-5 |
| Effort (how many files/lines change) | 30% | 1-5 (inverse: lower effort = higher score) |
| Risk (chance of breaking existing behavior) | 30% | 1-5 (inverse: lower risk = higher score) |

**False Positive Screening (from project-qc):**
- Auto-reject: complexity justified by domain requirements (e.g., state machines)
- Auto-reject: abstraction required by framework conventions
- Auto-reject: duplication across test files (acceptable)
- Downgrade: style preferences that don't affect maintainability

### worker-complexity.md

**Embedded criteria from:** `code-review` perspective 2 (Complexity & Readability)

**Checks:**
- Functions exceeding 80 lines
- Files exceeding 400 lines
- Nesting deeper than 3 levels
- Cyclomatic complexity (estimated by counting branches/loops)
- AI anti-pattern: "Abstraction Bloat" — elaborate class hierarchies where simpler approaches suffice
- AI anti-pattern: unnecessary scaffolding (factory/registry/strategy for single implementation)

**Output format per finding:**
```
[COMPLEXITY-N] file:line | severity: HIGH/MEDIUM/LOW
Description: What is complex and why it's unnecessary
Simpler alternative: Concrete suggestion with pseudocode
Estimated reduction: X lines → Y lines
```

### worker-dry.md

**Embedded criteria from:** `code-review` perspective 7 (DRY Violations)

**Checks:**
- Copy-pasted logic across files (near-duplicate detection)
- Duplicated constants or configuration
- Similar functions that should be unified
- AI anti-pattern: "Regeneration instead of reuse" — code generated fresh instead of importing existing utility
- AI anti-pattern: similar error handling blocks repeated across modules

**Output format per finding:**
```
[DRY-N] files: [list] | severity: HIGH/MEDIUM/LOW
Description: What is duplicated
Unification strategy: How to consolidate (extract function, shared constant, etc.)
Locations: file1:line, file2:line, ...
```

### worker-yagni.md

**Embedded criteria from:** `reviewing-plans` checks 1 (Scope Alignment) and 4 (Proportionality)

**Red flag phrases (from reviewing-plans):**
- "future-proof" / "extensible design"
- Interfaces/factories/registries for a single implementation
- Configuration options that are never changed from defaults
- Abstract base classes with only one concrete subclass

**Checks:**
- Premature abstraction: interface with single implementation
- Unused extension points: plugin systems, hook mechanisms never used
- Over-configuration: env vars / config options with only one value ever used
- AI anti-pattern: "Assumption Propagation" — architecture built on faulty premises
- AI anti-pattern: defensive code for impossible scenarios

**Output format per finding:**
```
[YAGNI-N] file:line | severity: HIGH/MEDIUM/LOW
Description: What is unnecessary and why
Evidence: Why this is YAGNI (single impl, never configured, etc.)
Simplification: What to remove or inline
```

### worker-dead-code.md

**Embedded criteria from:** `code-review` perspective 6 (Dead Code)

**Checks:**
- Unused imports
- Uncalled functions or methods (trace call graph)
- Commented-out code blocks
- Unreachable code paths (after return/raise/break)
- AI anti-pattern: "Dead Code Accumulation" — old implementations left after refactoring
- AI anti-pattern: orphaned helper functions from previous iterations
- Variables assigned but never read

**Output format per finding:**
```
[DEAD-N] file:line | severity: HIGH/MEDIUM/LOW
Description: What is dead and evidence it's unused
Confidence: HIGH (no references) / MEDIUM (only test references) / LOW (possible dynamic use)
Action: DELETE / VERIFY_THEN_DELETE / FLAG_FOR_REVIEW
```

### worker-coupling.md

**Embedded criteria from:** `code-review` perspectives 1 (Design Quality) and 5 (Changeability)

**Checks:**
- Circular import dependencies between modules
- God modules (one module imported by >60% of others)
- Shotgun surgery: changing one feature requires touching 5+ files
- Feature envy: module heavily uses another module's internals
- AI anti-pattern: "Context Blindness" — violating unwritten codebase conventions
- AI anti-pattern: tight coupling introduced by generated glue code
- Hidden dependencies: implicit ordering requirements, global state

**Output format per finding:**
```
[COUPLING-N] modules: [list] | severity: HIGH/MEDIUM/LOW
Description: What coupling exists and why it's problematic
Dependency direction: A → B (should be B → A, or neither)
Decoupling strategy: How to reduce coupling
```

### ai-anti-patterns.md (Shared Reference)

Catalog of AI coding agent mistakes, embedded into each worker prompt's context. Sourced from research:

**Category 1: Structural Over-engineering**
- Abstraction Bloat: unnecessary class hierarchies, factory/strategy for single use
- Scaffolding Excess: elaborate setup for simple operations
- Config Explosion: too many configuration options for simple behavior

**Category 2: Code Accumulation**
- Dead Code After Refactoring: old implementations left in place
- Regeneration Instead of Reuse: fresh code instead of importing existing
- Copy-Paste Amplification: 8.3% → 12.3% duplication increase (GitClear 2024)

**Category 3: Context Failures**
- Assumption Propagation: wrong assumption → entire feature built on it
- Convention Violation: ignoring project-specific patterns
- Lost-in-Middle: information ignored in long contexts

**Category 4: Quality Gaps**
- Error Handling Gaps: missing null checks, bare except (~2x human rate)
- Concurrency Misuse: async primitive misuse, incorrect ordering (~2x human rate)
- Readability Decay: naming inconsistency, formatting problems

**Sources:**
- [The 80% Problem in Agentic Coding (Addy Osmani)](https://addyo.substack.com/p/the-80-problem-in-agentic-coding)
- [AI vs Human Code Gen Report (CodeRabbit)](https://www.coderabbit.ai/blog/state-of-ai-vs-human-code-generation-report)
- [AI Coding Agents 2026: Coherence Through Orchestration (Mike Mason)](https://mikemason.ca/writing/ai-coding-agents-jan-2026/)
- [Stack Overflow: Bugs and Incidents with AI Coding Agents](https://stackoverflow.blog/2026/01/28/are-bugs-and-incidents-inevitable-with-ai-coding-agents)

## Data Flow

```
Input: Project root path (cwd)

Phase 1 Output (Architecture Snapshot):
  {
    modules: [{name, path, lines, imports, exports}],
    dependencies: [{from, to, type}],
    conventions: [pattern descriptions],
    entry_points: [paths],
    config_sources: [paths]
  }

Phase 2 Output (5 Worker Reports):
  Each worker returns: [{id, file, line, severity, description, suggestion, metadata}]

Phase 3 Output (Synthesized Findings):
  Deduplicated, scored, prioritized list of simplification opportunities
  Grouped by refactoring unit (related findings that should be addressed together)

Phase 4 Output (Refactoring Plan):
  docs/plans/YYYY-MM-DD-simplification-plan.md
  Format: writing-plans compatible (bite-sized tasks, exact file paths, verification steps)
  Verified by: /reviewing-plans
```

## Integration with Existing Skills

| Existing Skill | How Used |
|---|---|
| `dispatching-parallel-agents` | Pattern for Phase 2 parallel worker dispatch |
| `project-qc` | False positive screening + severity escalation patterns |
| `code-review` | Perspectives 1,2,5,6,7 criteria embedded in workers |
| `reviewing-plans` | Phase 4 plan verification (HARD GATE) |
| `writing-plans` | Output format for the refactoring plan |
| `plan-execution` | Downstream: executes the approved plan |

## Error Handling

- Worker subagent fails → log failure, continue with remaining workers, note gap in synthesis
- All workers fail → abort with error message
- Zero findings → report "no simplification opportunities found" (this is a valid outcome)
- reviewing-plans REJECT → revise plan based on reviewer feedback, re-submit (max 2 iterations)

## Testing Strategy (per writing-skills TDD)

**RED Phase:**
- Run subagent on a known over-engineered codebase WITHOUT skill
- Document: does it find real issues? Does it hallucinate? Does it over-report?

**GREEN Phase:**
- Write SKILL.md + worker templates
- Run same scenario WITH skill
- Verify: structured output, no false positives, actionable suggestions

**REFACTOR Phase:**
- Edge cases: empty project, single-file project, already clean project
- Verify false positive screening works (framework-required abstractions)
- Verify cross-perspective dedup (same issue found by 2+ workers)
