#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Function to creating loading bar
void loadingBar()
{
    // 0 - black background,
    // A - Green Foreground
    //system("color 0A");

    // Initialize char for printing
    // loading bar
    char a = 177, b = 219;

    printf("\n\n\n\n");
    printf("\n\n\n\n\t\t\t\t\tLoading...\n\n");
    printf("\t\t\t\t\t");

    // Print initial loading bar

    // Set the cursor again starting
    // point of loading bar
    printf("\r");
    printf("\t\t\t\t\t");

    // Print loading bar progress
    for (int i = 0; i < 26; i++) {
        printf("%c", b);

        // Sleep for 1 second
        sleep(1000);
    }
}

// Driver Code
int main()
{
    // Function Call
    loadingBar();
    return 0;
}

