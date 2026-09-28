/** @type {import('stylelint').Config} */
export default {
  extends: ['stylelint-config-standard'],
  rules: {
    // The stylesheet groups related rules without blank lines between them.
    'rule-empty-line-before': null,
    'comment-empty-line-before': null,
    // Focus rings for every control are one rule near the buttons, ahead of
    // the controls' own rules, so they stay in one place.
    'no-descending-specificity': null,
  },
};
