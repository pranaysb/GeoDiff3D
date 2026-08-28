import { Code, User, Mail } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="max-w-3xl mx-auto py-10">
      <h1 className="text-3xl font-bold mb-8">About GeoDiff3D</h1>
      
      <div className="space-y-6 text-zinc-300 leading-relaxed">
        <p>
          GeoDiff3D is an experimental computer vision system designed to investigate the integration of foundation diffusion models with classical and learned multi-view geometry.
        </p>
        <p>
          The project is built on the hypothesis that the strong semantic scene priors learned by 2D image diffusion models can effectively resolve the ambiguities present in sparse-view 3D reconstruction, specifically in textureless and highly-specular regions.
        </p>
      </div>

      <h2 className="text-xl font-semibold mt-12 mb-6">Contact & Links</h2>
      <div className="flex flex-col space-y-4">
        <a href="#" className="inline-flex items-center text-zinc-400 hover:text-emerald-400 transition-colors">
          <Code className="w-5 h-5 mr-3" />
          GitHub Repository
        </a>
        <a href="#" className="inline-flex items-center text-zinc-400 hover:text-emerald-400 transition-colors">
          <User className="w-5 h-5 mr-3" />
          LinkedIn
        </a>
        <a href="#" className="inline-flex items-center text-zinc-400 hover:text-emerald-400 transition-colors">
          <Mail className="w-5 h-5 mr-3" />
          Research Inquiries
        </a>
      </div>
    </div>
  );
}
