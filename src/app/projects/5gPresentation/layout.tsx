import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import FiveGProject from "./page";
const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Create Next App",
  description: "Generated by create next app",
};

export default function FiveGLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    
      <section>
        <FiveGProject></FiveGProject>
      </section>
      

  );
}
