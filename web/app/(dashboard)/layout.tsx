import NavBar from "@/components/NavBar";
import React from "react";
import SideBar from "@/components/SideBar";
import Footer from "@/components/footer";

const DashboardLayout = ({children}: { children: React.ReactNode }) => {
    return (
        <div className="h-full relative">
            <div className="hidden h-full md:flex md:w-72 md:flex-col md:fixed md:inset-y-0 z-[80] bg-gray-900">
                <SideBar/>
            </div>
            <main className="md:pl-72 bg-white dark:bg-gray-700">
                <NavBar/>
                {children}
                <Footer/>
            </main>
        </div>
    );
};

export default DashboardLayout;
