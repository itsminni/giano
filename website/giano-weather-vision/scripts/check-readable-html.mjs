import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const html = await readFile(
  new URL("../dist/client/index.html", import.meta.url),
  "utf8",
);
const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/)?.[1];
assert(main, "Missing readable main content in Pages HTML");
for (const section of ["home", "impatto", "storia", "guida"]) {
  assert(main.includes(`id="${section}"`), `Missing section: ${section}`);
  assert(html.includes(`href="#${section}"`), `Missing link: ${section}`);
}
assert(main.includes("giano-impute"), "Missing usage instructions");
assert(main.includes("Gabriele Mininni"), "Missing project credits");
assert(/<table\b/.test(main), "Missing results table");
assert(
  !/<(?:button|input|select|canvas)\b/.test(main) &&
    !main.includes("recharts-responsive-container"),
  "The readable page contains controls that require JavaScript",
);
console.log(
  "Pages HTML: project, results, story and guide readable without JavaScript.",
);
