import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
      colors: {
        'clay-red' : '#DD7E6B',
        'light-pink': '#EA9999',
        'light-orange': '#F9CB9C',
        'light-yellow': '#FFE599',
        'light-green': '#b6d7a8',
        'green-blue': '#a2c4c9',
        'light-blue': '#a4c1f4'

      }
    },
  },
  plugins: [],
};
export default config;
