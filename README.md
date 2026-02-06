# NN

This Project has lasted long enough where I've need to update this readme

This is my NN system/arch, running fully on the hands of numpy.
This is not based on any paper or talked about neural archeture, this is my own creation
Because of logically never existing until now, these are handwritten components; and we pray it works

This orginally came about from two choices:
- Floating point is cringe
- Error handling being possible to realize
    - Instead of just hallucinating an answer

Keeping in line with both of these limitations, we've fallen into a pattern based system.
This also limits what it can do organically, as pattern matching can't create anything that isn't from something else.

So now is the time where I explain how this works
- Each Stimulus is a bit, so a u64 contains 64 seperate stimuli
- The input, called a State, for each can scale beyond 64 bits due to only being bitwise
    - not affected by size limits for the most part, (I'm looking at you AVX)
    - a note is that both the positve and inverse is checked here, for what exists and what doesn't
- Each State is compared to several stored internal States, containing the States that is matching for
- Any matches found activate the corsponding emit Case which are combined to return as the output

- If the input can't be recovered by reversing this process, then we have a gap that must be filled
- The Diff(erence) is isolated, where it is handed off to begin the case generation
- The Diff is comapred against the existing cases to see if it exists but is masked off by the inverse to grow a case
- If it is not found, it will generate a new case for it to live in
    - Case generation does not rely on provided vales, as it checks for revcovery and not exact output


A lot is still required, however there are some notes:
I don't see this being used as a lanuage model, but it isn't impossible if one structures it correctly.
This remains untested, with the current test goal being to identify numbers from images; the worst case for it.

The feature list before release is as follows:
- Pray to frick the test works
- Maybe add the tinyest abstraction

Strech goals include:
- Porting to C + ASM (not C++, I want the memory management done myself)
- Creating a visualizer for case selection
- Attempting to do a LLM anyways, because surely it'll work
- Invest in alternate routes that might allow for the above
- Actually submitting a paper if this has grounds in usecases
    - Even if we can perfectly replicate other systems, the memory footprint can't reach GBs by design
    - That is quite the improvement regardless


Thank you for your continued interest in this project, it means the world to me
I won't say your witnessing history, but you are allowing me to continue putting it all into this
