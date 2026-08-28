import Link from 'next/link';
import { Beaker, UploadCloud, MonitorPlay, FileText, Info } from 'lucide-react';

export default function Navbar() {
  return (
    <nav className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <Link href="/" className="flex items-center space-x-2">
              <Beaker className="h-6 w-6 text-emerald-500" />
              <span className="text-xl font-bold tracking-tight text-zinc-100">GeoDiff<span className="text-emerald-500">3D</span></span>
            </Link>
            <div className="hidden sm:ml-10 sm:flex sm:space-x-8">
              <Link href="/reconstruct" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-zinc-300 hover:text-white">
                <UploadCloud className="h-4 w-4 mr-2" />
                Reconstruct
              </Link>
              <Link href="/compare" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-zinc-300 hover:text-white">
                <MonitorPlay className="h-4 w-4 mr-2" />
                Compare
              </Link>
              <Link href="/research" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-zinc-300 hover:text-white">
                <FileText className="h-4 w-4 mr-2" />
                Research
              </Link>
              <Link href="/about" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-zinc-300 hover:text-white">
                <Info className="h-4 w-4 mr-2" />
                About
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
