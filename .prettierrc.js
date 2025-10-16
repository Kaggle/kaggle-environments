module.exports = {
  overrides: [
    {
      // Target all JavaScript and TypeScript files
      files: ["*.js", "*.jsx", "*.ts", "*.tsx"],
      options: {
        trailingComma: "none",
        quoteProps: "preserve",
        semi: true,
        printWidth: 120
      }
    }
  ]
};
