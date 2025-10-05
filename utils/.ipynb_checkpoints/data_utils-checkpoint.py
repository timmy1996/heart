{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "da8b4c7c-d0b6-4a04-9d2c-654eaccd824e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def miss_val_perc(df):\n",
    "    miss_per_col  = df.isnull().sum()\n",
    "\n",
    "\n",
    "    mising_val_percent = np.round(100 * miss_per_col / len(df),2)\n",
    "    \n",
    "    \n",
    "    missing = pd.concat([miss_per_col, mising_val_percent], axis=1)\n",
    "    missing = missing.rename(\n",
    "            columns = {0 : 'Number of Missing Values', 1 : 'Percent of Total Values'}).sort_values('Percent of Total Values',ascending=False)\n",
    "    return missing\n",
    "    "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
