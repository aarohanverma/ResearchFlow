import { redirect } from "next/navigation";

/** Root redirects to feed for logged-in users or login for guests. */
export default function RootPage() {
  redirect("/feed");
}
