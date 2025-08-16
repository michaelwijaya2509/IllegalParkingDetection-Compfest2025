"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FiHome,
  FiCamera,
  FiBarChart2,
  FiSettings,
  FiMenu,
  FiX,
  FiMapPin,
  FiAlertTriangle,
  FiLogOut,
  FiChevronDown,
} from "react-icons/fi";
import Image from "next/image";
import Logo from "../../public/horus-ai.png";

const Navigation = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);
  const pathname = usePathname();
  const profileMenuRef = useRef<HTMLDivElement>(null);

  const navItems = [
    { href: "/", label: "Dashboard", icon: FiHome },
    { href: "/live-feed", label: "Live Feed", icon: FiCamera },
    { href: "/incidents", label: "Incidents", icon: FiAlertTriangle },
    { href: "/analytics", label: "Analytics", icon: FiBarChart2 },
    { href: "/newlocation", label: "Add New Location", icon: FiMapPin },
  ];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        profileMenuRef.current &&
        !profileMenuRef.current.contains(event.target as Node)
      ) {
        setIsProfileMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [profileMenuRef]);

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 px-4 sm:px-6 lg:px-8 bg-tile1 border-b border-gray-700 backdrop-blur-sm">
        <div className="max-w-8xl mx-auto">
          <div className="flex justify-between items-center h-20">
            <div className="flex items-center">
              <Link href="/" className="flex items-center space-x-3">
                <Image
                  src={Logo}
                  alt="Horus AI Logo"
                  width={60}
                  height={60}
                  className="h-40 w-40"
                  priority
                />
              </Link>
            </div>

            <div className="hidden md:flex items-center space-x-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-blue-600 text-white"
                        : "text-gray-300 hover:text-white hover:bg-tile2"
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </div>

            <div
              className="hidden md:flex items-center relative"
              ref={profileMenuRef}
            >
              <button
                onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
                className="flex items-center space-x-3 p-2 rounded-lg hover:bg-tile2 transition-colors"
              >
                <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center">
                  <span className="text-sm font-medium text-white">A</span>
                </div>
                <div className="text-sm text-left">
                  <p className="text-white font-medium">Admin</p>
                </div>
                <FiChevronDown
                  className={`w-4 h-4 text-gray-400 transition-transform ${
                    isProfileMenuOpen ? "rotate-180" : ""
                  }`}
                />
              </button>

              {isProfileMenuOpen && (
                <div className="absolute top-full right-0 mt-2 w-48 bg-tile2 border border-gray-700 rounded-lg shadow-lg py-1">
                  <Link
                    href="/settings"
                    onClick={() => setIsProfileMenuOpen(false)}
                    className="flex items-center w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white"
                  >
                    <FiSettings className="w-4 h-4 mr-3" />
                    Settings
                  </Link>
                </div>
              )}
            </div>

            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-300 hover:text-white hover:bg-tile2"
            >
              {isMobileMenuOpen ? (
                <FiX className="w-6 h-6" />
              ) : (
                <FiMenu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {isMobileMenuOpen && (
          <div className="md:hidden bg-tile2 border-t border-gray-700">
            <div className="px-2 pt-2 pb-3 space-y-1">
              {[
                ...navItems,
                { href: "/settings", label: "Settings", icon: FiSettings },
              ].map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center space-x-3 px-3 py-2 rounded-lg text-base font-medium transition-colors ${
                      isActive
                        ? "bg-blue-600 text-white"
                        : "text-gray-300 hover:text-white hover:bg-tile1"
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </div>

            <div className="border-t border-gray-700 px-2 pt-3 pb-3">
              <div className="flex items-center px-3 mb-3">
                <div className="w-10 h-10 bg-gray-600 rounded-full flex items-center justify-center mr-3">
                  <span className="text-sm font-medium text-white">A</span>
                </div>
                <div>
                  <p className="text-white font-medium">Admin</p>
                  <p className="text-gray-400 text-sm">Supervisor</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </nav>
    </>
  );
};

export default Navigation;
