"use client";
import Link from "next/link";
import gsap from 'gsap';
import { useState, useEffect, useRef } from 'react';
import EarthModel from '@/components/earth';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
const Typed = require('typed.js');

const pool = [
    "expanding the limitations of AI",
    "bringing AI to EVERYONE",
    "making the AI suitable for ALL",
    "breaking the barriers",
    "empowering the world",
    "shaping the future",
    "creating a better world",
];

export default function Earth() {
    const [mesh, setMesh] = useState<THREE.Mesh | undefined>();
    const t1 = gsap.timeline({ defaults: { duration: 1 } });
    const el = useRef(null);
    if (mesh) {
        t1.fromTo(mesh.scale, { x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 });
        t1.fromTo(".title", { opacity: 0 }, { opacity: 1 });
        t1.fromTo(".typed-fade-out", { opacity: 0 }, { opacity: 1 });
    }

    const [size, setSize] = useState({ width: 500, height: 500 });
    useEffect(() => {
        if (mesh) {
            mesh.scale.set(0, 0, 0);
        }
        const minSize = Math.min(window.innerWidth, window.innerHeight) * 0.8;
        setSize({ width: minSize, height: minSize });
        const handleResize = () => {
            setSize({ width: minSize, height: minSize });
        };
        window.addEventListener('resize', handleResize);
        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, [mesh]);

    useEffect(() => {
        const typed = new Typed(el.current, {
            strings: pool,
            typeSpeed: 80,
            backSpeed: 80,
            smartBackspace: true,
            startDelay: 2500,
            backDelay: 2000,
            shuffle: true,
            loop: true,
            loopCount: Infinity,
            showCursor: false
        });

        return () => {
            typed.destroy();
        };
    }, []);

    return (
        <>
            <section className="relative h_screen flex flex-col items-center justify-center py-0 px-3 bg-black w-full" >
                <div className="video-docker absolute top-0 left-0 w-full h-full overflow-hidden bulr starter-bg" >
                    <video className="min-w-full min-h-full absolute object-cover lg:opacity-20 opacity-30" autoPlay loop muted webkit-playsinline="true" playsInline x5-playsinline="true" id="bg_video">
                        <source src={'/bg.mp4'} type="video/mp4" />
                        video not supported
                    </video>
                </div>
                <div className='opacity-95 w-full'>
                    <div className='w-full justify-between lg:flex h-screen'>
                        <div className='w-full lg:w-1/2 lg:ml-16 my-auto mx-auto text-gray-200 title'>
                            <h1 className="text-8xl font-bold my-auto">
                                Grand Recs
                            </h1>
                            <h2 className=" text-3xl font-medium mt-8">
                                Together, We are <span ref={el}/>
                            </h2>
                            <h3 className="text-2xl mt-8 typed-fade-out underline">
                                <Link href="/dashboard" key="dashboard" passHref={true}>
                                    Enter the world &gt;
                                </Link>
                            </h3>
                        </div>
                        <div id="canvas-container" style={{ width: size.width, height: size.height }}
                            className='w-full lg:w-1/2 lg:mr-16 my-auto mx-auto'>
                            <Canvas camera={{
                                fov: 45,
                                aspect: size.width / size.height,
                                near: 0.1,
                                far: 1000,
                                position: [0, 0, 20]
                            }}>
                                <OrbitControls enableDamping={true} enablePan={false} enableZoom={false} autoRotate={true}
                                            autoRotateSpeed={1.1}/>
                                <pointLight color="#FFD1AB" intensity={1000} distance={30} position={[0, 10, 10]}/>
                                <ambientLight color="#fdfbd3" intensity={6}/>
                                <EarthModel setMesh={setMesh}/>
                            </Canvas>
                        </div>
                    </div>
                </div>
            </section>
        </>
    )
}