const path = require('path');

module.exports = {
  entry: './web.js',
  mode: 'production',
  // mode: 'development',
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist'),
    // module: true,
    // chunkFormat: "module",
  },
  experiments: {
    asyncWebAssembly: true,
    // outputModule: true,
  },
};
