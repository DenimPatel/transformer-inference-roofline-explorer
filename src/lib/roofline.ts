/**
 * Pure roofline math shared by the Interactive Lab and the Learn journey.
 *
 * All equations mirror the reference material in /reference/scaling-book/
 * (roofline.md + inference.md). Physics is unchanged from the original app —
 * this is an extraction, not a rewrite.
 */

export interface PhysicalUnits {
  flops: number;          // effective FLOPs/s (already scaled by precision)
  memBw: number;          // bytes/s
  memCap: number;         // bytes
  totalParams: number;    // count
  activeParams: number;   // count
  bytesPerParam: number;
  bytesPerToken: number;  // KV bytes per token
  hardwareRatio: number;  // FLOPs : byte (ridge point)
}

export interface RowInputs {
  totalParams: number;
  activeParams: number;
  bytesPerParam: number;
  contextLen: number;     // tokens
  bytesPerToken: number;  // KV bytes per token
  tdpWatts: number;
  hardwarePrice: number;
  priceKwh: number;
  pue: number;
  utilization: number;    // percent
  amortizationYears: number;
  isLiquid?: boolean;
}

export function physicalUnits(x: {
  flopsTera: number;
  memBwTera: number;
  memCapGb: number;
  totalParamsB: number;
  activeParamsB: number;
  bytesPerParam: number;
  bytesPerTokenKb: number;
}): PhysicalUnits {
  const precisionMultiplier = 2 / x.bytesPerParam;
  const flops = x.flopsTera * 1e12 * precisionMultiplier;
  const memBw = x.memBwTera * 1e12;
  const memCap = x.memCapGb * 1e9;
  const totalParams = x.totalParamsB * 1e9;
  const activeParams = x.activeParamsB * 1e9;
  const bytesPerToken = x.bytesPerTokenKb * 1024;
  return {
    flops,
    memBw,
    memCap,
    totalParams,
    activeParams,
    bytesPerParam: x.bytesPerParam,
    bytesPerToken,
    hardwareRatio: flops / memBw,
  };
}

/** One row of the latency/cost curve for a batch size B. */
export function computeRow(B: number, u: PhysicalUnits, x: RowInputs) {
  const tCompute = (B * u.activeParams * 2) / u.flops;
  const tWeightFetch = (u.totalParams * x.bytesPerParam) / u.memBw;
  const tKvFetch = (B * x.contextLen * u.bytesPerToken) / u.memBw;
  const tMemory = tWeightFetch + tKvFetch;
  const latency = Math.max(tCompute, tMemory);
  const throughput = B / latency;
  const isMemoryBound = tMemory > tCompute;
  const computeUtilization = isMemoryBound ? tCompute / tMemory : 1;
  const powerDrawW = isMemoryBound ? 0.5 * x.tdpWatts + 0.5 * x.tdpWatts * computeUtilization : x.tdpWatts;
  const joulesPerToken = powerDrawW * (latency / B);
  const costPerToken = latency / B;
  const effectivePue = x.isLiquid ? Math.min(x.pue, 1.05) : x.pue;
  const costElec = (joulesPerToken / 3600000) * effectivePue * x.priceKwh;
  const secondsAmortization = x.amortizationYears * 365 * 24 * 3600;
  const costHardware = (x.hardwarePrice / (secondsAmortization * (x.utilization / 100))) * (latency / B);
  const totalCostPerTokenUSD = costElec + costHardware;

  return {
    batchSize: B,
    tCompute: tCompute * 1000, // ms
    tMemory: tMemory * 1000,
    tWeightFetch: tWeightFetch * 1000,
    tKvFetch: tKvFetch * 1000,
    latency: latency * 1000, // ms
    throughput,
    costPerToken: costPerToken * 1000,
    joulesPerToken,
    powerDrawW,
    costElec1M: costElec * 1e6,
    costHardware1M: costHardware * 1e6,
    totalCost1M: totalCostPerTokenUSD * 1e6,
    isMemoryBound,
  };
}

export function logSpacedSizes(minExp = 0, maxExp = 16, step = 0.5): number[] {
  const sizes: number[] = [];
  for (let i = minExp; i <= maxExp; i += step) sizes.push(Math.round(Math.pow(2, i)));
  return Array.from(new Set(sizes)).filter((v) => v > 0);
}

/** Pope's balance point: where weight-read and compute time would be equal. */
export function optimalBatch(u: PhysicalUnits): number {
  return u.hardwareRatio * (u.totalParams / u.activeParams);
}

export function drainTimeMs(u: PhysicalUnits): number {
  return (u.memCap / u.memBw) * 1000;
}

export function contextCrossover(u: PhysicalUnits): number {
  return (u.activeParams * 2 * u.memBw) / (u.flops * u.bytesPerToken);
}

export function maxThroughput(u: PhysicalUnits): number {
  return u.flops / (u.activeParams * 2);
}

/** Arithmetic intensity of one decode step at batch B (FLOPs / bytes). */
export function currentIntensity(B: number, u: PhysicalUnits, x: RowInputs): number {
  return (B * u.activeParams * 2) / (u.totalParams * x.bytesPerParam + B * x.contextLen * u.bytesPerToken);
}

export function makeCurve(u: PhysicalUnits, x: RowInputs) {
  return logSpacedSizes().map((B) => computeRow(B, u, x));
}
