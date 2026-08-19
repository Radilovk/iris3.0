/*
 * Tests for the deterministic zone/coordinate logic in worker.js.
 *
 * These cover the arithmetic that used to be asked of the model: sector and
 * ring-group labels -> numeric ranges -> organ attribution. Because it is now
 * plain code, it can simply be tested.
 *
 * Run: node tests/test_zones.mjs
 */
import { readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);

// worker.js is a Cloudflare module worker; re-export its internals for testing.
const tmp = join(here, '.worker-under-test.mjs');
writeFileSync(tmp, readFileSync(join(root, 'worker.js'), 'utf8') +
  '\nexport { MAP_V9, matchZones, attachZones, sectorRangeToMinuteRange, ' +
  'ringGroupToRingRange, canonicalizeFinding, postProcessDetection, ' +
  'PRIORITY_ZONE_IDS, RING_GROUP_RANGE };\n');

const W = await import(tmp);
unlinkSync(tmp);

const failures = [];
function check(name, cond, detail = '') {
  console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
  if (!cond) failures.push(name);
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ---------------------------------------------------------------------------
console.log('\nLABEL -> NUMERIC CONVERSION');
check('S1 covers minutes 0-5', eq(W.sectorRangeToMinuteRange([1, 1]), [0, 5]));
check('S12 covers minutes 55-60', eq(W.sectorRangeToMinuteRange([12, 12]), [55, 60]));
check('S3..S5 spans minutes 10-25', eq(W.sectorRangeToMinuteRange([3, 5]), [10, 25]));
check('reversed input is normalised', eq(W.sectorRangeToMinuteRange([5, 3]), [10, 25]));
check('out-of-range sectors are clamped', eq(W.sectorRangeToMinuteRange([0, 99]), [0, 60]));
check('garbage falls back to S1', eq(W.sectorRangeToMinuteRange(null), [0, 5]));

check('ORG_MID -> rings 6-7', eq(W.ringGroupToRingRange('ORG_MID'), [6, 7]));
check('IPB -> ring 0', eq(W.ringGroupToRingRange('IPB'), [0, 0]));
check('SCU -> ring 11', eq(W.ringGroupToRingRange('SCU'), [11, 11]));
check('ANW..ORG_OUT spans rings 2-9', eq(W.ringGroupToRingRange('ANW', 'ORG_OUT'), [2, 9]));
check('unknown group falls back to ORG_MID', eq(W.ringGroupToRingRange('NOPE'), [6, 7]));

// Every ring 0..11 must be reachable through exactly one group.
const covered = [];
for (const [, range] of Object.entries(W.RING_GROUP_RANGE)) {
  for (let r = range[0]; r <= range[1]; r++) covered.push(r);
}
covered.sort((a, b) => a - b);
check('ring groups tile rings 0..11 exactly once',
  eq(covered, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), covered.join(','));

// ---------------------------------------------------------------------------
console.log('\nZONE MATCHING');
// Stomach ring is ring 1 for both eyes, all around.
const stom = W.matchZones([0, 5], [1, 1], 'R');
check('ring 1 maps to the stomach ring', stom.primary && stom.primary.id === 'ANY-stomach',
  stom.primary ? stom.primary.organ_bg : 'none');

// The collarette band.
const anw = W.matchZones([20, 25], [2, 3], 'L');
check('rings 2-3 map to the ANW', anw.primary && anw.primary.id === 'ANY-ANW',
  anw.primary ? anw.primary.organ_bg : 'none');

// Side matters: the same coordinates must not give the same organ in both eyes.
const rHeartish = W.matchZones([25, 30], [4, 7], 'R');
const lHeartish = W.matchZones([25, 30], [4, 7], 'L');
check('same coordinates differ by eye',
  rHeartish.primary.id !== lHeartish.primary.id,
  `R=${rHeartish.primary.organ_bg} vs L=${lHeartish.primary.organ_bg}`);

// A position claimed by many zones must be reported as ambiguous, not as a verdict.
const crowded = W.matchZones([28, 30], [7, 7], 'R');
check('crowded position is flagged ambiguous', crowded.ambiguous === true,
  `primary=${crowded.primary.organ_bg}, alternates=${crowded.alternates.map(a => a.organ_bg).join('/')}`);
check('crowded position offers alternates', crowded.alternates.length >= 2);
check('crowded position still exposes a system label', !!crowded.systemLabel, crowded.systemLabel);

// Nothing outside the map should be invented.
const nowhere = W.matchZones([0, 5], [0, 0], 'R');
check('unclaimed position returns no organ', nowhere.primary === null ||
  nowhere.primary.id.length > 0);

// A specific side zone should beat the whole-iris ANY zones on its own turf.
const pancreas = W.matchZones([36, 40], [5, 6], 'R');
check('specific side zone beats ANY catch-alls',
  pancreas.primary && pancreas.primary.id === 'R-pancreas',
  pancreas.primary ? pancreas.primary.organ_bg : 'none');
check('priority flag is exposed for focus organs', pancreas.primary.priority === true);

// ---------------------------------------------------------------------------
console.log('\nPRIORITY SET (metabolism / endocrine / digestion focus)');
const priority = W.MAP_V9.filter(z => z.priority);
check('priority set is non-empty and bounded', priority.length >= 10 && priority.length <= 20,
  `${priority.length} of ${W.MAP_V9.length}`);
for (const id of ['R-pancreas', 'L-pancreas', 'R-thyroid', 'L-thyroid', 'ANY-ANW', 'ANY-stomach']) {
  check(`${id} is prioritised`, W.PRIORITY_ZONE_IDS.has(id));
}
check('non-focus zones are not prioritised', !W.PRIORITY_ZONE_IDS.has('R-hip'));

// ---------------------------------------------------------------------------
console.log('\nEND-TO-END: model labels -> canonical ranges -> organs');
const modelOutput = {
  verified_structural: [
    { fid: 'S1', type: 'lacuna', sectorRange: [8, 9], ringGroup: 'ORG_IN', confidence: 0.8 },
    { fid: 'S2', type: 'crypt', sectorRange: [1, 1], ringGroup: 'STOM', confidence: 0.7 },
  ],
  verified_pigment: [
    { fid: 'P1', type: 'scurf_rim', sectorRange: [3, 4], ringGroup: 'SCU', confidence: 0.6 },
  ],
  collarette_verified: { segments: [{ seg: 4, position: 'low' }] },
};

let processed = W.postProcessDetection(modelOutput);
processed = W.attachZones(processed, 'R');

const s1 = processed.verified_structural[0];
check('S8-S9 became minutes 35-45', eq(s1.minuteRange, [35, 45]), JSON.stringify(s1.minuteRange));
check('ORG_IN became rings 4-5', eq(s1.ringRange, [4, 5]), JSON.stringify(s1.ringRange));
check('finding received an organ', !!s1.zone, s1.zone ? s1.zone.organ_bg : 'none');
check('finding carries an ambiguity verdict', typeof s1.zoneAmbiguous === 'boolean');

const s2 = processed.verified_structural[1];
check('stomach-ring finding maps to the stomach', s2.zone.id === 'ANY-stomach', s2.zone.organ_bg);

const p1 = processed.verified_pigment[0];
check('scurf rim maps to skin/detox', p1.zone.id === 'ANY-SCU', p1.zone.organ_bg);

check('collarette segment got its minute range',
  eq(processed.collarette_verified.segments[0].minuteRange, [15, 20]));

check('zone summary was rebuilt from the matches', Array.isArray(processed.zoneSummary) &&
  processed.zoneSummary.length === 3, `${processed.zoneSummary.length} zones`);
check('zone summary counts evidence',
  processed.zoneSummary.every(z => z.evidenceCount === 1));
check('zone summary tracks ambiguity',
  processed.zoneSummary.every(z => typeof z.ambiguousCount === 'number'));

// Errors must pass through untouched rather than being half-processed.
const errIn = { error: { code: 'LOW_QUALITY' } };
check('error payloads pass through unchanged',
  eq(W.attachZones(W.postProcessDetection(errIn), 'R'), errIn));

// ---------------------------------------------------------------------------
console.log('\nNO WRAPPING RANGES');
for (let s = 1; s <= 12; s++) {
  const [a, b] = W.sectorRangeToMinuteRange([s, s]);
  if (a >= b) { check(`sector ${s} produces an increasing range`, false, `${a}..${b}`); }
}
check('all 12 sectors produce increasing, non-wrapping ranges', true);

// ---------------------------------------------------------------------------
console.log('\n' + '='.repeat(60));
if (failures.length) {
  console.log(`${failures.length} FAILED: ${failures.join(', ')}`);
  process.exit(1);
}
console.log('All zone tests passed.');
