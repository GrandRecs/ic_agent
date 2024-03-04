import { LayoutDashboard, BrainCircuit, Database, Settings,  } from "lucide-react";

export const MAX_FREE_COUNTS = 5;

export const tools = [
  {
    label: "Dashboard",
    icon: LayoutDashboard,
    href: "/dashboard",
    color: "text-sky-500",
  },
  {
    label: "Personal Agent",
    icon: BrainCircuit,
    href: "/agent",
    color: "text-violet-500",
  },
  {
    label: "Dataset Collection",
    icon: Database,
    href: "/datasets",
    color: "text-violet-500",
  },
  {
    label: "Model Settings",
    icon: Settings,
    href: "/settings",
    color: "text-violet-500",
  },
];