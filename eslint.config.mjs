import js from '@eslint/js';
import stylistic from '@stylistic/eslint-plugin';
import { defineConfig, globalIgnores } from 'eslint/config';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default defineConfig(
  globalIgnores(['dist/', 'target/', 'test-results/', 'playwright-report/']),
  js.configs.recommended,
  stylistic.configs.customize({
    indent: 2,
    quotes: 'single',
    semi: true,
    braceStyle: '1tbs',
    commaDangle: 'always-multiline',
    arrowParens: true,
  }),
  {
    rules: {
      // Operators lead continued lines, except `=`.
      '@stylistic/operator-linebreak': ['error', 'before', { overrides: { '=': 'after' } }],
    },
  },
  {
    files: ['web/**/*.ts', 'tests/**/*.ts', 'playwright.config.ts'],
    extends: [tseslint.configs.recommendedTypeChecked],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.json', './tsconfig.worker.json', './tsconfig.tests.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // UI handlers are async and report their own failures.
      '@typescript-eslint/no-misused-promises': ['error', { checksVoidReturn: { arguments: false } }],
    },
  },
  {
    files: ['web/main/**/*.ts'],
    languageOptions: { globals: globals.browser },
  },
  {
    files: ['web/worker/**/*.ts'],
    languageOptions: { globals: globals.worker },
  },
  {
    // Tools run under Node and hand callbacks to the browser.
    files: ['tools/**/*.mjs', '*.mjs', 'tests/**/*.ts', 'playwright.config.ts'],
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
  },
);
