"use client";
import Navigation from "@/components/Navigation";
import { FiSave, FiBell, FiUser, FiGlobe } from "react-icons/fi";

export default function Settings() {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const formProps = Object.fromEntries(formData.entries());

    const settings = {
      ...formProps,
      emailNotif: formProps.emailNotif === "true",
    };

    console.log("Saving settings:", settings);
    alert("Settings saved! (Check console for data)");
  };

  return (
    <div className="min-h-screen bg-primary">
      <Navigation />
      <main className="pt-20 p-6 mt-10">
        <div className="max-w-3xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
            <p className="text-gray-400">
              Manage your account and notification preferences.
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="bg-tile1 border border-gray-700 rounded-lg p-8 space-y-8">
              <div>
                <h2 className="text-lg font-semibold text-white mb-4 border-b border-gray-700 pb-2 flex items-center">
                  <FiUser className="mr-3" /> Profile
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                  <div>
                    <label
                      htmlFor="fullName"
                      className="block text-sm font-medium text-gray-300 mb-2"
                    >
                      Full Name
                    </label>
                    <input
                      type="text"
                      name="fullName"
                      id="fullName"
                      defaultValue="Admin User"
                      className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="role"
                      className="block text-sm font-medium text-gray-300 mb-2"
                    >
                      Role
                    </label>
                    <input
                      type="text"
                      name="role"
                      id="role"
                      defaultValue="Administrator"
                      disabled
                      className="w-full bg-gray-700 border border-gray-600 rounded-md text-gray-400 px-4 py-2 cursor-not-allowed"
                    />
                  </div>
                </div>
              </div>

              <div>
                <h2 className="text-lg font-semibold text-white mb-4 border-b border-gray-700 pb-2 flex items-center">
                  <FiBell className="mr-3" /> Notifications
                </h2>
                <div className="space-y-4 mt-4">
                  <div className="flex items-center">
                    <input
                      id="emailNotif"
                      name="emailNotif"
                      type="checkbox"
                      value="true"
                      defaultChecked
                      className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 bg-gray-700"
                    />
                    <label
                      htmlFor="emailNotif"
                      className="ml-3 block text-sm font-medium text-gray-300"
                    >
                      Enable Email Notifications for Incidents
                    </label>
                  </div>
                  <div>
                    <label
                      htmlFor="emailAddress"
                      className="block text-sm font-medium text-gray-300 mb-2"
                    >
                      Recipient Email Address
                    </label>
                    <input
                      type="email"
                      name="emailAddress"
                      id="emailAddress"
                      placeholder="admin@example.com"
                      defaultValue="admin@horus.ai"
                      className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>

              <div>
                <h2 className="text-lg font-semibold text-white mb-4 border-b border-gray-700 pb-2 flex items-center">
                  <FiGlobe className="mr-3" /> General
                </h2>
                <div className="mt-4">
                  <label
                    htmlFor="timezone"
                    className="block text-sm font-medium text-gray-300 mb-2"
                  >
                    Timezone
                  </label>
                  <select
                    id="timezone"
                    name="timezone"
                    className="w-full bg-primary border border-gray-600 rounded-md text-white px-4 py-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option>Asia/Jakarta (GMT+7)</option>
                    <option>Asia/Singapore (GMT+8)</option>
                    <option>UTC</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="mt-6">
              <button
                type="submit"
                className="w-full flex items-center justify-center bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
              >
                <FiSave className="mr-2" />
                Save Settings
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
