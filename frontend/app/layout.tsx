import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Integrity Lab",
  description: "Full-stack plagiarism and AI detection system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

