const path = require('path');
const dotenv = require('dotenv');

// Manually load the .env file from the parent directory
dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

module.exports = {
  reactStrictMode: true,
  env: {
    PORT_FRONTEND: process.env.PORT_FRONTEND,
    PORT_BACKEND: process.env.PORT_BACKEND,
  },
};