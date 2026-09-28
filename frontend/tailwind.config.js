/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        slateInk: "#0F172A",
        navy: "#1E3A8A",
        blueAction: "#2563EB",
        panel: "#F8FAFC",
        app: "#F1F5F9",
        borderSoft: "#CBD5E1",
        borderStrong: "#94A3B8"
      },
      boxShadow: {
        card: "0 1px 3px rgba(15, 23, 42, 0.08)"
      }
    }
  },
  plugins: []
};
