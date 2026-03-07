import React, { useState } from 'react';
import { Box } from '@mui/material';
import AlignmentDisplayBlock from './AlignmentDisplayBlock.jsx'
import AlignmentForm from './AlignmentForm.jsx'


const AlignerContentBlock = () => {
    const [refText, setRefText] = useState('');
    const [compText, setCompText] = useState('');
    const [algo, setAlgo] = useState("alma");
    const [alignmentResult, setAlignmentResult] = useState(null);
    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <AlignmentForm
                setRefText={setRefText}
                setCompText={setCompText}
                setAlgo={setAlgo}
                setAlignmentResult={setAlignmentResult}
            />
            <AlignmentDisplayBlock
                refText={refText}
                compText={compText}
                result={alignmentResult}
            />
        </Box>
    );
};

export default AlignerContentBlock;
