import js from '@eslint/js';
import stylistic from '@stylistic/eslint-plugin';
import { defineConfig, globalIgnores } from 'eslint/config';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default defineConfig(
  globalIgnores(['dist/', 'target/']),
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
      // Operators lead a continued line, but an assignment's `=` stays at
      // the end of the line it starts on.
      '@stylistic/operator-linebreak': ['error', 'before', { overrides: { '=': 'after' } }],
    },
  },
  {
    files: ['web/**/*.ts'],
    extends: [tseslint.configs.recommendedTypeChecked],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.json', './tsconfig.worker.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // UI handlers are async on purpose and report their own failures;
      // the check stays on for every other place a promise is misused.
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
    // Tools run under Node, and hand callbacks to the browser they drive.
    files: ['tools/**/*.mjs', '*.mjs'],
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
  },
);
