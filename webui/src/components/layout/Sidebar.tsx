import { NavLink } from "react-router-dom"
import {
  LayoutDashboard,
  BarChart3,
  ScrollText,
  Settings,
  KeyRound,
  Filter,
  Zap,
} from "lucide-react"
import { cn } from "@/lib/utils"

const navItems = [
  { to: "/ui", icon: LayoutDashboard, label: "Dashboard", end: true },
  { to: "/ui/quota", icon: BarChart3, label: "Quota" },
  { to: "/ui/logs", icon: ScrollText, label: "Logs" },
  { to: "/ui/credentials", icon: KeyRound, label: "Credentials" },
  { to: "/ui/models", icon: Filter, label: "Models" },
  { to: "/ui/settings", icon: Settings, label: "Settings" },
]

export function Sidebar() {
  return (
    <aside className="hidden md:flex md:w-56 lg:w-64 flex-col border-r bg-card">
      <div className="flex items-center gap-2 px-4 h-14 border-b">
        <Zap className="h-5 w-5 text-primary" />
        <span className="font-semibold text-sm">LLM Proxy</span>
      </div>
      <nav className="flex-1 px-2 py-3 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              )
            }
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
