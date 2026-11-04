import { expect, test } from "bun:test";
import { resolve } from "node:path";
import { ESLint } from "eslint";

const eslint = new ESLint({ cwd: resolve(import.meta.dir, "..") });

test("lint includes TypeScript source and excludes generated website files", async () => {
  for (const path of [
    "src/components/giano/Story.tsx",
    "tests/site.test.ts",
    "vite.config.ts",
  ]) {
    expect(await eslint.isPathIgnored(path)).toBe(false);
  }
  for (const path of [
    "src/generated/catalog.ts",
    "src/routeTree.gen.ts",
    "public/data/generated.ts",
    ".output/server/index.ts",
    "dist/client/generated.ts",
  ]) {
    expect(await eslint.isPathIgnored(path)).toBe(true);
  }
});

test("the upgraded linter still catches conditional Hooks in TSX", async () => {
  const [result] = await eslint.lintText(
    `import { useState } from "react";
export function Example({ enabled }: { enabled: boolean }) {
  if (enabled) useState(0);
  return <div />;
}
`,
    { filePath: "src/lint-regression.tsx" },
  );
  expect(result?.messages.map((message) => message.ruleId)).toContain(
    "react-hooks/rules-of-hooks",
  );
});

test("the upgraded parser still applies TypeScript rules", async () => {
  const [result] = await eslint.lintText("export const value: any = 1;\n", {
    filePath: "src/lint-regression.ts",
  });
  expect(result?.messages.map((message) => message.ruleId)).toContain(
    "@typescript-eslint/no-explicit-any",
  );
});
