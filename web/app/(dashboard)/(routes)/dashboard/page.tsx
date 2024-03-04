import { Card } from '@/components/ui/card';

export default function HomePage() {
    return (
        <div className='h-screen'>
            <div className="mb-8 space-y-4 ">
                <h2 className="text-2xl md:text-4xl font-bold text-center">
                    LLM Assistants
                </h2>
                <p className="text-muted-foreground font-light text-sm md:text-lg text-center">
                    Tests the limits of Compact Language Models
                </p>
            </div>

            <div className="flex justify-center">
                <Card className="p-4 md:p-8 lg:p-12 space-y-4 max-w-lg w-full">
                    <h3 className="text-lg md:text-xl font-bold text-center">
                        Acknowledgment and Consent
                    </h3>
                    <p className="text-sm md:text-base text-center">
                        Our tools and models are designed to test the boundaries of current methodologies and provide
                        innovative solutions. Please note that the outputs from our models are experimental and should
                        not be taken as definitive advice or solutions. By using our tools, you acknowledge that we are
                        not responsible for the results produced and their interpretation.
                    </p>
                </Card>
            </div>
        </div>
    );
}