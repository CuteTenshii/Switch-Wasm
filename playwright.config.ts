import { defineConfig, devices } from '@playwright/test';

/* Page tests, in a real browser.

   Locally they run against the dev server, which serves `make wasm`'s dev
   build; in CI against `vite preview`, which serves the `dist` that
   `make assets` built. Either way the core has to be built first. */

const port = 8200;

export default defineConfig({
  testDir: 'tests/e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'list' : 'line',
  use: {
    baseURL: `http://localhost:${port}/`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: process.env.CI
      ? `bun run preview --port ${port} --strictPort`
      : `bun run dev --port ${port} --strictPort`,
    url: `http://localhost:${port}/`,
    reuseExistingServer: !process.env.CI,
  },
});
