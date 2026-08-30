// A small inline sample corpus (one "document" per line, like names.txt) used to seed
// the lab's training textarea before the user loads the full reference file or pastes
// their own. Every character is within microgpt.ts's DEFAULT_ALPHABET.
export const SAMPLE_CORPUS = [
  'olivia',
  'liam',
  'emma',
  'noah',
  'ava',
  'sophia',
  'lucas',
  'mia',
  'ethan',
  'harper',
  'ava',
  'amelia',
  'benjamin',
  'ella',
  'mason',
  'scarlett',
  'logan',
  'grace',
  'aiden',
  'chloe',
  'wyatt',
  'zoey',
  'carter',
  'lily',
  'maya',
  'caleb',
  'nora',
  'daniel',
  'hannah',
  'oliver',
];

// Imported raw from reference/names.txt via vite, split into lines (all lowercase a-z).
export const rawNames = `ada
lucy
edna
max
olly
nora
luke
elle
aria
milo
zayn
rory
kian
owen
ezra
jade
maria
theo
wren
boon
gaby
mira
eli
adam
rose
hugo
clara
felix
lena
tess
ivy
jude
maya
owen
julie
hans
dina
cleo
beck
root
elio
nina
owls
anne
perry
anouk
bryn
ezri
kyla
zuri
rex
`;

export function sampleNames(count: number, source: string): string[] {
  const lines = source.split('\n').map((s) => s.trim()).filter(Boolean);
  if (lines.length === 0) return [];
  // deterministic-ish pick across the spectrum so the button gives variety
  const step = Math.max(1, Math.floor(lines.length / count));
  const picked: string[] = [];
  for (let i = 0; i < count && i * step < lines.length; i++) picked.push(lines[i * step]);
  return picked;
}