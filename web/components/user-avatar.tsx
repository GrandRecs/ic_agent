import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

export const UserAvatar = () => {
    return (
        <Avatar className="h-8 w-8">
            <AvatarImage/>
            <AvatarFallback>
                You
            </AvatarFallback>
        </Avatar>
    );
};