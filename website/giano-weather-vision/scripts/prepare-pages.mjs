import { createHash } from "node:crypto";
import { readFile, readdir, stat, unlink, writeFile } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync, gunzipSync } from "node:zlib";

const output = fileURLToPath(new URL("../dist/client/", import.meta.url));
const data = resolve(output, "data/giano");
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
await stat(resolve(output, "index.html"));
const sourceManifest = await readFile(resolve(data, "manifest.json"));
const manifest = JSON.parse(sourceManifest);
let count = 0;

// Compress only the generated copy.
for (const [path, expected] of Object.entries(manifest.files)) {
  const file = resolve(data, path);
  if (!file.startsWith(`${data}${sep}`))
    throw new Error(`Invalid asset path: ${path}`);
  const bytes = await readFile(file);
  if (hash(bytes) !== expected)
    throw new Error(`Asset checksum mismatch: ${path}`);
  if (!path.startsWith("histories/") || !path.endsWith(".json")) continue;
  const compressed = gzipSync(bytes, { level: 9 });
  if (!gunzipSync(compressed).equals(bytes))
    throw new Error(`Compression failed: ${path}`);
  await writeFile(`${file}.gz`, compressed);
  await unlink(file);
  delete manifest.files[path];
  manifest.files[`${path}.gz`] = hash(compressed);
  count++;
}
manifest.source_manifest_sha256 = hash(sourceManifest);
manifest.history_transport = "gzip";
await writeFile(resolve(data, "manifest.json"), JSON.stringify(manifest));
await writeFile(resolve(output, ".nojekyll"), "");

async function size(directory) {
  let total = 0;
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = resolve(directory, entry.name);
    if (entry.isSymbolicLink()) throw new Error(`Unexpected symlink: ${file}`);
    total += entry.isDirectory() ? await size(file) : (await stat(file)).size;
  }
  return total;
}
const bytes = await size(output);
if (bytes >= 1_000_000_000)
  throw new Error(`Pages output exceeds 1 GB: ${bytes} bytes`);
console.log(
  `GitHub Pages: ${count} compressed history files, ${(bytes / 1_000_000).toFixed(1)} MB. Output: dist/client`,
);
