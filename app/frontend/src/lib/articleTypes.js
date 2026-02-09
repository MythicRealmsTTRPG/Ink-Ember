"import {
  FileText, Building2, User, Globe2, Shield, Sparkles,
  Mountain, Package, Users, Church, Dna, Car, Home,
  AlertTriangle, Swords, FileStack, Languages, Gem,
  Crown, ScrollText, Scale, Map, Briefcase, BookOpen,
  Wand2, Cpu, PartyPopper, ClipboardList, Layers
} from 'lucide-react';

export const articleTypes = {
  generic: {
    id: 'generic',
    label: 'Generic',
    icon: FileText,
    color: 'text-gray-500',
    bgColor: 'bg-gray-500/10',
    category: 'meta'
  },
  building: {
    id: 'building',
    label: 'Building',
    icon: Building2,
    color: 'text-amber-500',
    bgColor: 'bg-amber-500/10',
    category: 'world'
  },
  character: {
    id: 'character',
    label: 'Character',
    icon: User,
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
    category: 'characters'
  },
  country: {
    id: 'country',
    label: 'Country',
    icon: Globe2,
    color: 'text-emerald-500',
    bgColor: 'bg-emerald-500/10',
    category: 'civilization'
  },
  military: {
    id: 'military',
    label: 'Military',
    icon: Shield,
    color: 'text-red-500',
    bgColor: 'bg-red-500/10',
    category: 'civilization'
  },
  gods_deities: {
    id: 'gods_deities',
    label: 'Gods / Deities',
    icon: Sparkles,
    color: 'text-yellow-500',
    bgColor: 'bg-yellow-500/10',
    category: 'world'
  },
  geography: {
    id: 'geography',
    label: 'Geography',
    icon: Mountain,
    color: 'text-green-600',
    bgColor: 'bg-green-600/10',
    category: 'world'
  },
  item: {
    id: 'item',
    label: 'Item',
    icon: Package,
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
    category: 'items'
  },
  organization: {
    id: 'organization',
    label: 'Organization',
    icon: Users,
    color: 'text-indigo-500',
    bgColor: 'bg-indigo-500/10',
    category: 'civilization'
  },
  religion: {
    id: 'religion',
    label: 'Religion',
    icon: Church,
    color: 'text-violet-500',
    bgColor: 'bg-violet-500/10',
    category: 'world'
  },
  species: {
    id: 'species',
    label: 'Species',
    icon: Dna,
    color: 'text-pink-500',
    bgColor: 'bg-pink-500/10',
    category: 'world'
  },
  vehicle: {
    id: 'vehicle',
    label: 'Vehicle',
    icon: Car,
    color: 'text-cyan-500',
    bgColor: 'bg-cyan-500/10',
    category: 'items'
  },
  settlement: {
    id: 'settlement',
    label: 'Settlement',
    icon: Home,
    color: 'text-orange-500',
    bgColor: 'bg-orange-500/10',
    category: 'civilization'
  },
  condition: {
    id: 'condition',
    label: 'Condition',
    icon: AlertTriangle,
    color: 'text-rose-500',
    bgColor: 'bg-rose-500/10',
    category: 'characters'
  },
  conflict: {
    id: 'conflict',
    label: 'Conflict',
    icon: Swords,
    color: 'text-red-600',
    bgColor: 'bg-red-600/10',
    category: 'characters'
  },
  document: {
    id: 'document',
    label: 'Document',
    icon: FileStack,
    color: 'text-stone-500',
    bgColor: 'bg-stone-500/10',
    category: 'narrative'
  },
  culture_ethnicity: {
    id: 'culture_ethnicity',
    label: 'Culture / Ethnicity',
    icon: Users,
    color: 'text-teal-500',
    bgColor: 'bg-teal-500/10',
    category: 'world'
  },
  language: {
    id: 'language',
    label: 'Language',
    icon: Languages,
    color: 'text-sky-500',
    bgColor: 'bg-sky-500/10',
    category: 'world'
  },
  material: {
    id: 'material',
    label: 'Material',
    icon: Gem,
    color: 'text-slate-500',
    bgColor: 'bg-slate-500/10',
    category: 'world'
  },
  military_formation: {
    id: 'military_formation',
    label: 'Military Formation',
    icon: Shield,
    color: 'text-red-400',
    bgColor: 'bg-red-400/10',
    category: 'civilization'
  },
  myth: {
    id: 'myth',
    label: 'Myth',
    icon: ScrollText,
    color: 'text-amber-600',
    bgColor: 'bg-amber-600/10',
    category: 'world'
  },
  natural_law: {
    id: 'natural_law',
    label: 'Natural Law',
    icon: Scale,
    color: 'text-lime-500',
    bgColor: 'bg-lime-500/10',
    category: 'world'
  },
  plot: {
    id: 'plot',
    label: 'Plot',
    icon: Map,
    color: 'text-fuchsia-500',
    bgColor: 'bg-fuchsia-500/10',
    category: 'narrative'
  },
  profession: {
    id: 'profession',
    label: 'Profession',
    icon: Briefcase,
    color: 'text-zinc-500',
    bgColor: 'bg-zinc-500/10',
    category: 'civilization'
  },
  prose: {
    id: 'prose',
    label: 'Prose',
    icon: BookOpen,
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-600/10',
    category: 'narrative'
  },
  title: {
    id: 'title',
    label: 'Title',
    icon: Crown,
    color: 'text-yellow-600',
    bgColor: 'bg-yellow-600/10',
    category: 'civilization'
  },
  spell: {
    id: 'spell',
    label: 'Spell',
    icon: Wand2,
    color: 'text-violet-600',
    bgColor: 'bg-violet-600/10',
    category: 'items'
  },
  technology: {
    id: 'technology',
    label: 'Technology',
    icon: Cpu,
    color: 'text-blue-400',
    bgColor: 'bg-blue-400/10',
    category: 'world'
  },
  tradition: {
    id: 'tradition',
    label: 'Tradition',
    icon: PartyPopper,
    color: 'text-pink-400',
    bgColor: 'bg-pink-400/10',
    category: 'world'
  },
  session_report: {
    id: 'session_report',
    label: 'Session Report',
    icon: ClipboardList,
    color: 'text-indigo-400',
    bgColor: 'bg-indigo-400/10',
    category: 'narrative'
  }
};

export const articleCategories = {
  world: {
    id: 'world',
    label: 'World & Setting',
    icon: Globe2,
    types: ['geography', 'natural_law', 'material', 'technology', 'species', 'culture_ethnicity', 'language', 'religion', 'gods_deities', 'myth', 'tradition', 'building']
  },
  civilization: {
    id: 'civilization',
    label: 'Civilization & Power',
    icon: Crown,
    types: ['country', 'settlement', 'organization', 'military', 'military_formation', 'profession', 'title']
  },
  characters: {
    id: 'characters',
    label: 'Characters & Actors',
    icon: User,
    types: ['character', 'condition', 'conflict']
  },
  narrative: {
    id: 'narrative',
    label: 'Narrative & Canon',
    icon: BookOpen,
    types: ['plot', 'prose', 'document', 'session_report']
  },
  items: {
    id: 'items',
    label: 'Items & Systems',
    icon: Package,
    types: ['item', 'spell', 'vehicle']
  },
  meta: {
    id: 'meta',
    label: 'Meta & Utility',
    icon: Layers,
    types: ['generic']
  }
};

export const getArticleType = (typeId) => {
  return articleTypes[typeId] || articleTypes.generic;
};

export const getArticlesByCategory = (categoryId) => {
  const category = articleCategories[categoryId];
  if (!category) return [];
  return category.types.map(typeId => articleTypes[typeId]).filter(Boolean);
};
"