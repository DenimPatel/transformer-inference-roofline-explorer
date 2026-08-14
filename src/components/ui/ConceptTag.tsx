import InfoPopover from './InfoPopover';

interface ConceptTagProps {
  id: string;
  asButton?: boolean;
}

/** Small pill that opens the concept popover for `id`. The whole pill is clickable. */
export default function ConceptTag({ id }: ConceptTagProps) {
  return (
    <InfoPopover
      conceptId={id}
      iconSize={12}
      label={`Learn about ${id.replace(/-/g, ' ')}`}
      className="glass-chip px-2.5 py-1 text-[10.5px] font-medium text-slate-600 hover:bg-white/90"
    >
      <span className="capitalize">{id.replace(/-/g, ' ')}</span>
    </InfoPopover>
  );
}

export { ConceptTag };
