"use client";

import { BrainCircuit } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import React from "react";

const routes = [
    {
        label: "Hack the LLM Assistant Bot",
        icon: BrainCircuit,
        href: "/agent",
        color: "text-violet-500",
    },
];

const SideBar = () => {
    const pathName = usePathname();
    return (
        <div className="space-y-4 py-4 flex flex-col h-full bg-[#111827] text-white">
            <div className="px-3 py-2 flex-1">
                <Link href="/dashboard" className="flex items-center pl-3 mb-14">
                    <div className="relative w-8 h-8 mr-4">
                        <Image fill alt="logo" sizes="(max-width: 600px) 100vw, (max-width: 1200px) 50vw, 800px" src="/gr.png"/>
                    </div>
                    <h1 className="text-2xl font-bold">GrandRec</h1>
                </Link>
                <div className="space-y-1">
                    {routes.map((route) => (
                        <Link
                            href={route.href}
                            key={route.href}
                            className={`text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white ${pathName === route.href ? "bg-white/10" : ""} hover:bg-white/10 rounded-lg transition`}
                        >
                            <div className="flex item-center flex-1">
                                <route.icon className="h-5 w-5 mr-3"/>
                                {route.label}
                            </div>
                        </Link>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default SideBar;
