import { cn } from "@/lib/utils";

export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("skeleton rounded-lg", className)}
      {...props}
    />
  );
}

export function PaperCardSkeleton() {
  return (
    <div className="rounded-2xl border border-gray-800/60 bg-gray-900/40 p-5 space-y-3">
      <div className="flex gap-2">
        <Skeleton className="h-5 w-14" />
        <Skeleton className="h-5 w-20" />
      </div>
      <Skeleton className="h-5 w-full" />
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="h-4 w-1/2 opacity-60" />
      <Skeleton className="h-10 w-full opacity-40" />
      <div className="flex gap-4 pt-1">
        <Skeleton className="h-3 w-28" />
        <Skeleton className="h-3 w-24" />
      </div>
    </div>
  );
}
