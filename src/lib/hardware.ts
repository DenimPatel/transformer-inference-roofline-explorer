export interface HardwareProfile {
  vendor: string;
  id: string;
  tflops: number;
  memBw: number;
  capacity: number;
  arch: string;
  bytesPerParam: number; // 2 for FP16/BF16, 1 for FP8, 0.5 for FP4
  tdp: number; // Watts
  price: number; // Hardware cost (USD)
  color: string;
}

export const HARDWARE_PROFILES: HardwareProfile[] = [
  { vendor: 'NVIDIA', id: 'H100 SXM5', tflops: 989, memBw: 3.35, capacity: 80, arch: 'Hopper (GPU)', bytesPerParam: 2, tdp: 700, price: 30000, color: '#76b900' },
  { vendor: 'NVIDIA', id: 'H200 SXM', tflops: 989, memBw: 4.80, capacity: 141, arch: 'Hopper (HBM3e)', bytesPerParam: 2, tdp: 700, price: 40000, color: '#3f8c00' },
  { vendor: 'NVIDIA', id: 'B200 (Blackwell)', tflops: 9000, memBw: 8.00, capacity: 192, arch: 'Blackwell (GPU)', bytesPerParam: 1, tdp: 1000, price: 45000, color: '#b9ff42' },
  { vendor: 'NVIDIA', id: 'Rubin GPU (R100)', tflops: 50000, memBw: 22.00, capacity: 288, arch: 'Rubin (GPU)', bytesPerParam: 0.5, tdp: 2000, price: 60000, color: '#00af50' },
  { vendor: 'NVIDIA', id: 'Vera CPU', tflops: 400, memBw: 1.20, capacity: 1500, arch: 'Vera (CPU)', bytesPerParam: 2, tdp: 2300, price: 20000, color: '#a0f000' },
  { vendor: 'NVIDIA', id: 'L40S', tflops: 366, memBw: 0.86, capacity: 48, arch: 'Ada Lovelace', bytesPerParam: 2, tdp: 350, price: 10000, color: '#00af50' },
  { vendor: 'Google', id: 'TPU v5p', tflops: 459, memBw: 2.58, capacity: 95, arch: 'TPU (Pod-scale)', bytesPerParam: 2, tdp: 450, price: 15000, color: '#ea4335' },
  { vendor: 'Google', id: 'TPU v6e (Trillium)', tflops: 918, memBw: 1.64, capacity: 32, arch: 'TPU (Inference)', bytesPerParam: 2, tdp: 300, price: 10000, color: '#fbbc04' },
  { vendor: 'Google', id: 'TPU v7 (Ironwood)', tflops: 4614, memBw: 7.37, capacity: 192, arch: 'TPU (Ironwood)', bytesPerParam: 1, tdp: 600, price: 25000, color: '#4285f4' },
  { vendor: 'Google', id: 'TPU v8i (Inference)', tflops: 10100, memBw: 8.60, capacity: 288, arch: 'TPU (Inference)', bytesPerParam: 0.5, tdp: 750, price: 30000, color: '#fbbc04' },
  { vendor: 'Google', id: 'TPU v8t (Training)', tflops: 12600, memBw: 6.50, capacity: 216, arch: 'TPU (Training)', bytesPerParam: 0.5, tdp: 900, price: 35000, color: '#34a853' },
  { vendor: 'AMD', id: 'Instinct MI300X', tflops: 1307, memBw: 5.30, capacity: 192, arch: 'CDNA 3 (GPU)', bytesPerParam: 2, tdp: 750, price: 15000, color: '#ed1c24' },
  { vendor: 'AMD', id: 'Instinct MI325X', tflops: 1307, memBw: 6.00, capacity: 256, arch: 'CDNA 3 (HBM3e)', bytesPerParam: 2, tdp: 1000, price: 20000, color: '#a00e12' },
  { vendor: 'AWS', id: 'Trainium2 (Trn2)', tflops: 1300, memBw: 2.90, capacity: 96, arch: 'NeuronCore-v2', bytesPerParam: 1, tdp: 600, price: 15000, color: '#ff9900' },
  { vendor: 'AWS', id: 'Inferentia2 (Inf2)', tflops: 190, memBw: 0.82, capacity: 32, arch: 'NeuronCore-v2', bytesPerParam: 2, tdp: 185, price: 5000, color: '#232f3e' },
  { vendor: 'Groq', id: 'Groq LPU v1', tflops: 18.8, memBw: 80.0, capacity: 0.23, arch: 'LPU (Inference)', bytesPerParam: 2, tdp: 185, price: 20000, color: '#f3522a' },
  { vendor: 'Groq', id: 'Groq 3 LPU', tflops: 500, memBw: 80.0, capacity: 0.5, arch: 'LPU (Inference)', bytesPerParam: 1, tdp: 250, price: 30000, color: '#ff5400' },
  { vendor: 'Hybrid', id: 'Groq + Rubin Pipeline', tflops: 50000, memBw: 80.0, capacity: 288.5, arch: 'Hybrid (LPU + GPU)', bytesPerParam: 0.5, tdp: 2250, price: 90000, color: '#f59e0b' },
  { vendor: 'SambaNova', id: 'SN40L (RDU)', tflops: 638, memBw: 2.0, capacity: 64, arch: 'RDU (3-tier)', bytesPerParam: 2, tdp: 400, price: 25000, color: '#652d90' },
];
