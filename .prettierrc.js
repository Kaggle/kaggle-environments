module.exports = {
    overrides: [
        {
            // Target all JavaScript and TypeScript files
            files: ['*.js', '*.jsx', '*.ts', '*.tsx'],
            options: {
                tabWidth: 4,
                singleQuote: true,
                trailingComma: 'none',
                quoteProps: 'preserve',
                semi: true,
                printWidth: 120
            }
        }
    ]
};
