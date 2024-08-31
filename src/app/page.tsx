import Image from "next/image";
import csgo from './cards/CSGO_CARD.svg'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
       <span>
        <object data='./cards/CSGO_CARD.svg' />
      </span> 
    </main>
  );
}


