
Action: file_editor create /app/frontend/src/pages/LandingPage.js --file-text "import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { useTheme, themes } from '@/contexts/ThemeContext';
import {
  Flame, BookOpen, History, Map, GitFork, Variable,
  StickyNote, CheckSquare, ScrollText, Calendar,
  ArrowRight, Sparkles
} from 'lucide-react';

const features = [
  {
    icon: BookOpen,
    title: 'Wiki-Style Articles',
    description: 'Familiar article pages with sections, links, and infoboxes. 28+ article types for every aspect of your world.'
  },
  {
    icon: History,
    title: 'Timelines & Calendars',
    description: 'Multiple calendars per world. Events that span timelines. Custom calendar systems for your setting.'
  },
  {
    icon: ScrollText,
    title: 'Chronicles',
    description: 'Campaign logs, story arcs, historical records. Built on articles and events, not isolated text.'
  },
  {
    icon: Variable,
    title: 'Variables',
    description: 'World states, canon layers, visibility controls. Support for \"what-if\" and alternate timelines.'
  },
  {
    icon: Map,
    title: 'Maps',
    description: 'Link regions, settlements, and routes directly to articles. Narrative context, not just geography.'
  },
  {
    icon: GitFork,
    title: 'Family Trees',
    description: 'Character relationships, family lineages. Diplomatic relations between organizations and nations.'
  }
];

export default function LandingPage() {
  const { theme, setTheme, allThemes } = useTheme();

  return (
    <div className=\"min-h-screen\">
      {/* Hero Section */}
      <section className=\"relative overflow-hidden\">
        {/* Background */}
        <div className=\"absolute inset-0 bg-gradient-to-b from-background via-background to-card\" />
        <div className=\"absolute inset-0 opacity-30\">
          <div className=\"absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl\" />
          <div className=\"absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/10 rounded-full blur-3xl\" />
        </div>

        <div className=\"relative max-w-6xl mx-auto px-4 py-20 md:py-32\">
          {/* Logo */}
          <div className=\"flex items-center justify-center gap-3 mb-8\">
            <Flame className=\"w-12 h-12 md:w-16 md:h-16 text-primary\" />
            <h1 className=\"font-heading text-4xl md:text-6xl font-bold\">
              Ink & Ember
            </h1>
          </div>

          {/* Tagline */}
          <p className=\"text-center text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-4\">
            A worldbuilding and narrative management platform for writers and tabletop creators who work on long-term, complex projects.
          </p>
          <p className=\"text-center text-base text-muted-foreground/80 max-w-xl mx-auto mb-12\">
            Ink for what is written. Ember for what continues to burn.
          </p>

          {/* CTA */}
          <div className=\"flex flex-col sm:flex-row items-center justify-center gap-4\">
            <Link to=\"/worlds\">
              <Button size=\"lg\" className=\"gap-2 text-base px-8\" data-testid=\"get-started-btn\">
                Get Started
                <ArrowRight className=\"w-4 h-4\" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Philosophy Section */}
      <section className=\"py-16 md:py-24 bg-card\">
        <div className=\"max-w-6xl mx-auto px-4\">
          <h2 className=\"font-heading text-2xl md:text-3xl font-semibold text-center mb-12\">
            Built for Projects Measured in Years
          </h2>

          <div className=\"grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto\">
            <div className=\"space-y-2\">
              <h3 className=\"font-heading text-lg font-medium flex items-center gap-2\">
                <Sparkles className=\"w-5 h-5 text-primary\" />
                Wiki-First Presentation
              </h3>
              <p className=\"text-muted-foreground\">
                Lore should be readable, browsable, and linkable like a traditional wiki.
              </p>
            </div>
            <div className=\"space-y-2\">
              <h3 className=\"font-heading text-lg font-medium flex items-center gap-2\">
                <Sparkles className=\"w-5 h-5 text-primary\" />
                Structured Under the Hood
              </h3>
              <p className=\"text-muted-foreground\">
                Articles are typed, field-driven entries that can be queried, related, and reused.
              </p>
            </div>
            <div className=\"space-y-2\">
              <h3 className=\"font-heading text-lg font-medium flex items-center gap-2\">
                <Sparkles className=\"w-5 h-5 text-primary\" />
                Creator Ownership
              </h3>
              <p className=\"text-muted-foreground\">
                Your work should not be trapped. Export and long-term access are core design goals.
              </p>
            </div>
            <div className=\"space-y-2\">
              <h3 className=\"font-heading text-lg font-medium flex items-center gap-2\">
                <Sparkles className=\"w-5 h-5 text-primary\" />
                Scales with Complexity
              </h3>
              <p className=\"text-muted-foreground\">
                Whether running a single campaign or managing decades of shared universe history.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className=\"py-16 md:py-24\">
        <div className=\"max-w-6xl mx-auto px-4\">
          <h2 className=\"font-heading text-2xl md:text-3xl font-semibold text-center mb-12\">
            Core Features
          </h2>

          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6\">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className=\"p-6 rounded-lg border border-border bg-card card-hover animate-slide-up\"
                  style={{ animationDelay: `${index * 0.05}s` }}
                  data-testid={`feature-card-${index}`}
                >
                  <Icon className=\"w-8 h-8 text-primary mb-4\" />
                  <h3 className=\"font-heading text-lg font-medium mb-2\">{feature.title}</h3>
                  <p className=\"text-sm text-muted-foreground\">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Theme Preview */}
      <section className=\"py-16 md:py-24 bg-card\">
        <div className=\"max-w-6xl mx-auto px-4\">
          <h2 className=\"font-heading text-2xl md:text-3xl font-semibold text-center mb-4\">
            Choose Your Aesthetic
          </h2>
          <p className=\"text-center text-muted-foreground mb-12 max-w-xl mx-auto\">
            Four distinct themes to match your world's tone. Switch anytime.
          </p>

          <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto\">
            {Object.values(allThemes).map((t) => (
              <button
                key={t.id}
                onClick={() => setTheme(t.id)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  theme === t.id
                    ? 'border-primary ring-2 ring-primary/20'
                    : 'border-border hover:border-primary/50'
                }`}
                data-testid={`theme-btn-${t.id}`}
              >
                <div
                  className=\"w-full h-12 rounded mb-3 flex items-center justify-center\"
                  style={{ backgroundColor: t.preview.bg }}
                >
                  <div
                    className=\"w-6 h-6 rounded-full\"
                    style={{ backgroundColor: t.preview.primary }}
                  />
                </div>
                <div className=\"text-sm font-medium\">{t.name}</div>
                <div className=\"text-xs text-muted-foreground\">{t.type}</div>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Footer CTA */}
      <section className=\"py-16 md:py-24\">
        <div className=\"max-w-2xl mx-auto px-4 text-center\">
          <Flame className=\"w-12 h-12 text-primary mx-auto mb-6\" />
          <h2 className=\"font-heading text-2xl md:text-3xl font-semibold mb-4\">
            Start Building Your World
          </h2>
          <p className=\"text-muted-foreground mb-8\">
            Designed for people who are not \"done worldbuilding\" — and never plan to be.
          </p>
          <Link to=\"/worlds\">
            <Button size=\"lg\" className=\"gap-2\" data-testid=\"footer-get-started-btn\">
              Enter Ink & Ember
              <ArrowRight className=\"w-4 h-4\" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className=\"py-8 border-t border-border\">
        <div className=\"max-w-6xl mx-auto px-4 text-center text-sm text-muted-foreground\">
          <p>Ink & Ember — Worldbuilding Without Limits</p>
        </div>
      </footer>
    </div>
  );
}
"
Observation: Create successful: /app/frontend/src/pages/LandingPage.js