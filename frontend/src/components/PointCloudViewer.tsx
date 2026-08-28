"use client";

import React, { Suspense, useEffect, useMemo, useRef } from 'react';
import { Canvas, useLoader, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

interface PointCloudProps {
  url: string;
  pointSize: number;
}

/**
 * Reconstructed point clouds are not guaranteed to be centered near the
 * origin -- VGGT's coordinate frame depends on the input cameras, so a scene
 * can end up centered anywhere. A fixed camera position (the previous
 * behavior) renders a black screen whenever the cloud sits outside its view
 * frustum. This frames the camera on the geometry's actual bounding sphere
 * once it loads.
 */
function FrameOnLoad({ geometry }: { geometry: THREE.BufferGeometry }) {
  const { camera, controls } = useThree() as unknown as {
    camera: THREE.PerspectiveCamera;
    controls: { target: THREE.Vector3; update: () => void } | null;
  };

  useEffect(() => {
    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (!sphere || !isFinite(sphere.radius) || sphere.radius === 0) return;

    const distance = sphere.radius / Math.sin((Math.PI * camera.fov) / 360) * 1.2;
    camera.position.set(
      sphere.center.x,
      sphere.center.y,
      sphere.center.z + distance
    );
    camera.near = Math.max(distance / 100, 0.01);
    camera.far = distance * 100;
    camera.updateProjectionMatrix();

    if (controls) {
      controls.target.copy(sphere.center);
      controls.update();
    }
  }, [geometry, camera, controls]);

  return null;
}

function PLYModel({ url, pointSize }: PointCloudProps) {
  // Same ngrok free-tier interstitial bypass as lib/api.ts -- PLYLoader does
  // its own fetch outside that module, so it needs the header set directly.
  const geometry = useLoader(PLYLoader, url, (loader) => {
    loader.setRequestHeader({ "ngrok-skip-browser-warning": "true" });
  });

  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: geometry.hasAttribute('color'),
      sizeAttenuation: false,
    });
  }, [geometry, pointSize]);

  return (
    <>
      <FrameOnLoad geometry={geometry} />
      <points geometry={geometry} material={material} />
    </>
  );
}

export default function PointCloudViewer({ url, pointSize = 2 }: { url: string, pointSize?: number }) {
  const controlsRef = useRef(null);
  return (
    <div className="w-full h-full bg-zinc-950 rounded-lg overflow-hidden border border-zinc-800">
      <Canvas camera={{ position: [0, 0, 5], fov: 50 }}>
        <color attach="background" args={['#09090b']} />
        <ambientLight intensity={1} />
        <Suspense fallback={null}>
          <PLYModel url={url} pointSize={pointSize} />
        </Suspense>
        <OrbitControls ref={controlsRef} makeDefault />
      </Canvas>
    </div>
  );
}
