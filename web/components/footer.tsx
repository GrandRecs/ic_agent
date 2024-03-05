import Image from 'next/image';

export default function Footer() {
    return (
        <footer className="bg-white rounded-lg shadow dark:bg-gray-900 m-4">
            <div className="w-full max-w-screen-xl mx-auto p-4 md:py-8">
                <div className="sm:flex sm:items-center sm:justify-between">
                    <a href="/dashboard" className="flex items-center mb-4 sm:mb-0 space-x-3 rtl:space-x-reverse">
                        <div className="relative w-10 h-10 mr-4">
                            <Image width={500} height={500} alt="logo" src="/gr.png"/>
                        </div>
                        <h1 className="text-2xl font-bold">GrandRec</h1>
                    </a>
                    <ul className="flex flex-wrap items-center mb-6 text-sm font-medium text-gray-500 sm:mb-0 dark:text-gray-400">
                        <li>
                            <a href="https://github.com/GrandRecs/ic_agent/blob/main/README.md" className="hover:underline me-4 md:me-6">About</a>
                        </li>
                        <li>
                            <a href="https://github.com/GrandRecs/ic_agent/blob/main/LICENSE" className="hover:underline me-4 md:me-6">Licensing</a>
                        </li>
                        <li>
                            <a href="https://github.com/GrandRecs/ic_agent" className="hover:underline">Github</a>
                        </li>
                    </ul>
                </div>
                <hr className="my-6 border-gray-200 sm:mx-auto dark:border-gray-700 lg:my-8" />
                <span className="block text-sm text-gray-500 sm:text-center dark:text-gray-400">© {new Date().getFullYear()} GrandRec. All Rights Reserved. <br /> Assembled with <a href="https://nextjs.org/blog/next-14">Next.js 14</a></span>
            </div>
        </footer>
    )
}
