# Limitations, privacy and scope

Stated up front, because a system that forecasts urban problems invites more confidence than this
one has earned.

## Scope

UrbanSense is a **research and portfolio project**. It does not control city infrastructure, traffic
signals or any real-world system, and it is not designed to. Its outputs are analyses, not
instructions.

v1 covers **traffic only**. Flood and waste are designed for — the schema and configuration support
them — but not implemented. The architecture is deliberately deeper than it is wide
(SPEC section 46).

## Privacy

- **Public, sample and synthetic data only.** Nothing here depends on private data.
- **Zone-level aggregates.** No individual trips, vehicles or people.
- **No personal data of any kind**: no phone numbers, addresses, names, faces or device
  identifiers. No facial recognition and no individual tracking (SPEC section 45).
- `data/sample/` is committed to git, so anything placed there must be synthetic or public by
  construction.

The privacy limitation worth naming explicitly: aggregation protects privacy only when the zones are
large enough. A "zone" containing one building is not an aggregate. Any location registry added
later needs a minimum-population or minimum-area floor, and that check does not exist yet.

## Analytical limitations

**Correlation is not causation.** Explanations report which features *contributed* to a prediction.
That rainfall contributed 24% to a congestion forecast does not establish that rainfall caused the
congestion (SPEC section 27).

**Interventions are observational.** When a recommended action is followed and the outcome improves,
the system records both. It does not claim the action caused the improvement — there is no control
group, and the conditions that prompted the intervention are themselves correlated with the outcome
(SPEC section 36).

**Patterns can be artefacts of the data.** A pattern that strengthens may reflect a change in
sensor coverage rather than a change in the city. Pattern evolution tracking makes this visible; it
cannot distinguish the two on its own.

**Synthetic streaming is not reality.** The simulator demonstrates adaptive behaviour under
scenarios we constructed. Performance under synthetic drift is evidence about the *mechanism*, not
a forecast of accuracy on a real city's data.

**Document extraction is imperfect by assumption.** Hence confidence scores, the
`EXTRACTED_FACT`/`INFERRED` distinction and the human-review path. The guard against inventing
numbers is structural, but it bounds the failure rather than eliminating it: an extractor can still
misread a number that *is* present in the source.

## Engineering limitations (as of Phase 9)

v1 is feature-complete, so these are real limitations of working code rather than notes about
unwritten code.

### Not implemented

- **Document ingestion** (SPEC sections 5, 34, 35). `document_intelligence/` is a stub. The schema
  side is real -- document provenance is required and auditable, `INFERRED` extractions cannot be
  born `VALID`, and the unified layer has a `document_view` -- but nothing parses a PDF or a
  spreadsheet. Everything the paragraph above says about extraction confidence describes a design,
  not a measurement.
- **Human review UI** (SPEC section 35). Suspicious records are written to `quarantine/review/`
  with the fields a reviewer would need. There is no interface and nothing resolves a review.
- **Intervention tracking** (SPEC section 36). `recommendations/` is a stub. Nothing recommends an
  action or scores one, so the observational caveat above is likewise a design note.
- **Flood and waste.** Traffic is the only configured problem type.

### Data and evaluation

- **The data is synthetic and self-authored.** The generator and the detectors share an author, so
  the data cannot surprise them the way a real feed would. Every metric in this repository is a
  simulation result. See the README's [what is synthetic](../README.md#what-is-synthetic-and-what-that-costs).
- **One planted drift type.** A single clean multiplicative level shift. No gradual drift, no
  seasonal regime change, no sensor replacement -- which is the limitation most likely to be hiding
  the adaptive arm's real advantage, since Phase 8 found calendar retraining matched it here.
- **Thin protected slices.** The Friday-evening windows contribute 8-10 rows per four-week
  evaluation. Their per-slice figures move a lot between seeds and should not be read as precisely
  as the overall numbers.
- **Five seeds.** A spread from five datasets is itself imprecise. The Phase 8 "tie" means *not
  distinguished*, not *equal*.
- **One fixed calibrator in the experiment.** Fitted once from the shared starting champion and
  never refitted, so the Brier column compares regressors rather than complete serving systems.
- **The location registry has no size floor.** Aggregation protects privacy only when zones are
  large enough; a "zone" containing one building is not an aggregate. The registry validates
  `location_id` against a known list, but nothing enforces a minimum population or area.

### Serving and operations

- **No authentication anywhere.** Both the API and the dashboard are unauthenticated, which is why
  they bind `127.0.0.1` by default and say so in `/health`. There is no access control, no audit
  log of who read what, and no rate limiting.
- **No write path.** Corrections enter through the pipeline, never over HTTP. This is deliberate,
  but it means the review workflow above cannot be completed through the running services.
- **Artifacts are cached for the life of the API process.** A re-export needs a restart, or
  `--rebuild` on the way up.
- **Filters are applied in Python over whole artifacts.** Fine at 59,000 rows; this is not a
  database, and an order of magnitude more data would want one.
- **The quarantine queue has no retention policy.** Rejected payloads are kept indefinitely so
  rejections stay auditable. A real deployment would need an expiry, and that is a privacy
  consideration as much as a storage one.
- **The full test suite cannot run in one process on a modest machine.** It is split in two; the
  experiment suite generates and replays several datasets and exhausts memory alongside the rest.
- **Docker is unverified.** The `Dockerfile` and `docker-compose.yml` were written and are checked
  by tests for parse and posture, but no image has been built or run by the author.
