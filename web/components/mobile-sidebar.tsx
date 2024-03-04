"use client";

import React, { useEffect, useState } from "react";
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "./ui/sheet";
import SideBar from "./SideBar";

const MobileSidebar = () => {
    const [isMounted, setIsMounted] = useState(false);

    useEffect(() => {
        setIsMounted(true);
    }, []);

    if (!isMounted) {
        return null;
    }
    return (
        <Sheet>
            <SheetTrigger>
                <div className="md:hidden">
                    <Menu/>
                </div>
            </SheetTrigger>
            <SheetContent side="left" className="p-0">
                <SideBar/>
            </SheetContent>
        </Sheet>
    );
};

export default MobileSidebar;
