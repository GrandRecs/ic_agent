import React from "react";
import { headers } from 'next/headers'

import MobileSidebar from "./mobile-sidebar";

const NavBar = () => {
    const headersList = headers()
    const sn = headersList.get('sn') ?? '';
    const utorid = headersList.get('utorid') ?? '';
    const email = headersList.get('http_mail') ?? '';
    return (
        <div className="flex items-center p-4">
            <MobileSidebar/>
        </div>
    );
};

export default NavBar;
