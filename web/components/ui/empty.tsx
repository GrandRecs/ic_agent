interface EmptyProps {
    label: string;
}

export const Empty = ({
                          label
                      }: EmptyProps) => {
    return (
        <div className="h-full p-20 flex flex-col items-center justify-center">
            <p className="text-muted-foreground text-sm text-center">
                {label}
            </p>
        </div>
    );
};