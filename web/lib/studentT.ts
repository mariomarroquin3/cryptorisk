// Student-t density helpers -- mirrors the UNIT-VARIANCE convention in
// models/_dist.py::student_t_z (rescale the raw-t quantile/density by
// sqrt((nu-2)/nu) so `vol` below is a real standard deviation).

// Lanczos approximation, g=7 n=9 -- standard, accurate to ~1e-10 for the
// nu/2, (nu+1)/2 arguments used here (nu typically in [2, 30]).
const LANCZOS_G = 7;
const LANCZOS_P = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

function gamma(z: number): number {
  if (z < 0.5) return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
  const zz = z - 1;
  let x = LANCZOS_P[0];
  for (let i = 1; i < LANCZOS_G + 2; i++) x += LANCZOS_P[i] / (zz + i);
  const t = zz + LANCZOS_G + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, zz + 0.5) * Math.exp(-t) * x;
}

/** Density of a scaled (real std = `vol`) Student-t with `nu` degrees of
 * freedom, at return level `r`, mean 0. */
export function scaledStudentTDensity(r: number, vol: number, nu: number): number {
  const s = Math.sqrt((nu - 2) / nu);
  const scale = vol * s;
  const z = r / scale;
  const coef = gamma((nu + 1) / 2) / (Math.sqrt(nu * Math.PI) * gamma(nu / 2));
  return (coef * Math.pow(1 + (z * z) / nu, -(nu + 1) / 2)) / scale;
}
