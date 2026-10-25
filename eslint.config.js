import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    files: ['src/**/*.{ts,tsx}', 'tests/**/*.ts'],
    extends: [tseslint.configs.recommended],
    rules: {
      '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
      'no-console': 'off',
    },
  },
  {
    ignores: ['dist/', 'public/opencv/', 'node_modules/'],
  },
);
