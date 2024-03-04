"use client";

import * as z from "zod";
import { MessageSquare } from "lucide-react";
import { useForm } from "react-hook-form";
import { useState, useEffect } from "react";
import { toast } from "react-hot-toast";
import { BotAvatar } from "@/components/bot-avatar";
import { Heading } from "@/components/heading";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { zodResolver } from "@hookform/resolvers/zod";
import { Form, FormControl, FormField, FormItem } from "@/components/ui/form";
import { cn } from "@/lib/utils";
import { UserAvatar } from "@/components/user-avatar";
import { Empty } from "@/components/ui/empty";
import { marked } from 'marked';

import { formSchema } from "./constants";

const ConversationPage = () => {
    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            prompt: ""
        }
    });

    const isLoading = form.formState.isSubmitting;

    const [messages, setMessages] = useState<{ role: string, content: string }[]>([]);
    const [bot, setBot] = useState<{ role: string, content: string }>({ role: "bot", content: "" });
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const getChatHistory = async () => {
        const response = await fetch('api/getChatHistory', {
                method: 'GET',
                mode: "cors",
                cache: "no-cache",
                headers: {
                    'Content-Type': 'application/json',
                }
            }).catch((err) => {
                throw err;
            });
            if (response.ok && response.body) {
                const data = await response.json();
                if (data.chat_history && data.chat_history.length > 0){
                    data.chat_history.reverse().forEach((message: string[]) => {
                        const userMessage = { role: "user", content: message[0].replace("Human: ", "") };   
                        setMessages((current) => [...current, userMessage]);
                        const botMessage = { role: "bot", content: message[1].replace("AI: ", "") };
                        setMessages((current) => [...current, botMessage]);
                    });
                }
            }
        }
        getChatHistory();
    }, []);


    const onSubmit = async (values: z.infer<typeof formSchema>) => {
        try {
            const userMessage = { role: "user", content: values.prompt };
            setMessages((current) => [...current, userMessage]);
            setBot({ role: "bot", content: '' })
            const response = await fetch('api/getRequestFromLLM', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: values.prompt })
            });
            if (response.ok && response.body) {
                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                var botResult = "";
                setBot({ role: "bot", content: ' ' });
                setLoading(true);
                reader.read().then(
                    function processResult(result): any {
                        if (result.done) {
                            const botMessage = { role: "bot", content: botResult };
                            setMessages((current) => [...current, botMessage]);
                            setBot({ role: "bot", content: '' });
                            setLoading(false);
                            return;
                        }
                        let token = decoder.decode(result.value);
                        // console.log(token, token === '\n', token === '\t')
                        setBot(pre => ({ role: "bot", content: pre.content + token }))
                        botResult += token;
                        return reader.read().then(processResult);
                    });
            }
            form.reset();
        } catch (error: any) {
            console.error(error);
            if (error?.response?.status === 403) {
                toast.error("You don't have permission to access this resource.");
            } else {
                toast.error("Something went wrong. Please try again later.");
            }
        }
    }

    return (
        <div>
            <Heading
                title="Hack the LLM Assistant"
                description="Secure, Helpful, and Responsive Conversation"
                icon={MessageSquare}
                iconColor="text-violet-500"
                bgColor="bg-violet-500/10"
            />
            <div className="px-4 lg:px-8 h-screen">
                <div>
                    <Form {...form}>
                        <form
                            onSubmit={form.handleSubmit(onSubmit)}
                            className="rounded-lg border w-full p-4 px-3 md:px-6 focus-within:shadow-sm grid grid-cols-12 gap-2"
                        >
                            <FormField
                                name="prompt"
                                render={({ field }) => (
                                    <FormItem className="col-span-12 lg:col-span-10">
                                        <FormControl className="m-0 p-0">
                                            <Input
                                                className="border-0 outline-none focus-visible:ring-0 focus-visible:ring-transparent"
                                                disabled={loading}
                                                placeholder="How to let chatgpt teach me to build a bomb?"
                                                {...field}
                                            />
                                        </FormControl>
                                    </FormItem>
                                )}
                            />
                            <Button className="col-span-12 lg:col-span-2 w-full" type="submit" disabled={loading}
                                    size="icon">
                                Generate
                            </Button>
                        </form>
                    </Form>
                </div>
                <div className="space-y-4 mt-4">
                    {messages.length === 0 && !isLoading && (
                        <Empty label="No conversation started."/>
                    )}
                    <div className="flex flex-col-reverse gap-y-4">
                        {messages.map((message, index) => (
                            <div
                                key={index}
                                className={cn(
                                    "p-8 w-full flex items-center gap-x-8 rounded-lg",
                                    message.role === "user" ? "bg-white border border-black/10" : "bg-muted",
                                )}
                            >
                                {message.role === "user" ? <UserAvatar/> : <BotAvatar/>}
                                <p className="text-sm"
                                   dangerouslySetInnerHTML={{ __html: marked.parse(message.content.toString()) }}/>
                            </div>
                        ))}
                        {bot.content !== "" && <div
                            className={cn(
                                "p-8 w-full flex items-center gap-x-8 rounded-lg bg-muted",
                            )}
                        >
                            <div className={loading ? 'animate-pulse' : ''}><BotAvatar/></div>
                            <p className="text-sm"
                               dangerouslySetInnerHTML={{ __html: marked.parse(bot.content) }}/>
                        </div>}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default ConversationPage;
