module.exports = {
  overrides: [
    {
      // Target all JavaScript and TypeScript files
      files: ['*.js', '*.jsx', '*.ts', '*.tsx'],
      options: {
        tabWidth: 2,
        singleQuote: true,
        trailingComma: 'preserve',
        quoteProps: 'preserve',
        semi: true,
        printWidth: 120
      }
    }
  ]
};
