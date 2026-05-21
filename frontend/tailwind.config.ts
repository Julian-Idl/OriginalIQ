import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#18202f",
        panel: "#ffffff",
        line: "#dfe5ee",
        brand: "#1d6f9f",
      },
    },
  },
  plugins: [],
} satisfies Config;

